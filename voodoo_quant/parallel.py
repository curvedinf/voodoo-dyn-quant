"""True tensor parallelism (TP) for the Voodoo trainer.

Launched under ``torchrun`` (RANK/WORLD_SIZE/LOCAL_RANK), each rank patches the
model so it holds only 1/N of every parallelizable weight:

* MLP ``gate_proj``/``up_proj``   -> column-parallel (output rows sharded)
* MLP ``down_proj``               -> row-parallel (input cols sharded, all-reduce sum)
* attention ``q/k/v_proj``        -> column-parallel by heads
* attention ``o_proj``            -> row-parallel
* SSM ``in_proj_qkv``             -> column-parallel in the grouped [q|k|v] layout
  (the rank-local output is the concatenation of this rank's q, k and v head
  slices, matching the depthwise conv channel order)
* SSM ``in_proj_z/b/a``, ``dt_bias``/``A_log``, ``conv1d`` -> v-head sharded
* SSM ``out_proj``                -> row-parallel
* ``embed_tokens``                -> vocabulary-row sharded (masked lookup + all-reduce)
* ``lm_head``                     -> vocabulary-column sharded (per-rank logits;
  the loss combines them exactly via a log-sum-exp across ranks)

Parallel layers subclass ``nn.Linear``/``nn.Embedding`` so the existing
MixedQuant replacement walkers pick them up unchanged; they carry a
``tp_shard_spec`` (ShardSpec) plus a ``tp_full_weight`` reference to the full
CPU (usually mmap'd) weight so candidates are quantized/cached under the FULL
tensor name and then sliced to the rank-local shape (bit-identical to the
single-GPU cache).

The adapters also handle the Q8_0 teacher's ``Q8Linear``/``Q8Embedding``
modules (they hold ``qweight`` qbytes buffers), slicing the quantized bytes
instead of a bf16 weight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _TPState:
    """Global TP state; rank/world are 0/1 when TP is not enabled."""

    enabled = False
    rank = 0
    world_size = 1
    local_rank = 0
    group = None
    device: torch.device | None = None


TP = _TPState()


def tp_init(backend: str = "nccl", timeout_min: int = 240) -> bool:
    """Initialize the TP process group from torchrun's env variables.

    Returns True when TP is active (WORLD_SIZE > 1 and RANK present).  Each rank
    binds to its LOCAL_RANK GPU before init_process_group.  The generous
    default timeout matters: candidate-cache loading / teacher build can take
    well over an hour before the first collective.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world <= 1 or "RANK" not in os.environ:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=timeout_min),
        )
    TP.enabled = True
    TP.rank = dist.get_rank()
    TP.world_size = dist.get_world_size()
    TP.local_rank = local_rank
    TP.group = dist.group.WORLD
    TP.device = torch.device("cuda", local_rank)
    print(
        f"[tp] rank {TP.rank}/{TP.world_size} on {TP.device} (backend={dist.get_backend()})",
        flush=True,
    )
    return True


def tp_finalize():
    if TP.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        TP.enabled = False


def init_tp(backend: str | None = None, timeout_min: int = 240) -> tuple[int, int]:
    """Initialize the TP process group from torchrun's ``env://`` variables.

    Backend defaults to NCCL/RCCL when GPUs are present and gloo otherwise,
    so the same call works for GPU training and CPU-only smoke tests.  Under a
    plain (non-torchrun) launch TP stays disabled and ``(0, 1)`` is returned;
    every helper and parallel layer then degenerates to single-process
    behavior, so shared code paths need no ``if TP.enabled`` guards.

    Returns ``(rank, world_size)``.
    """
    if backend is None:
        backend = (
            "nccl"
            if torch.cuda.is_available() and dist.is_nccl_available()
            else "gloo"
        )
    tp_init(backend=backend, timeout_min=timeout_min)
    return TP.rank, TP.world_size


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """Alias of :func:`tp_all_reduce_sum`.

    Returns the reduced tensor (a new tensor in TP mode; ``t`` itself when TP
    is disabled) -- the input is never mutated in place.
    """
    return tp_all_reduce_sum(t)


def barrier() -> None:
    """Synchronize all TP ranks (no-op when TP is disabled)."""
    if TP.enabled and dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    return TP.rank == 0


class _AllReduceSum(Function):
    """Differentiable all-reduce SUM.

    Forward: every rank ends up with y = sum_r x_r.
    Backward: returns grad_out unchanged.  On rank r the autograd graph only
    connects y back to x_r (the other ranks' contributions arrive as constants
    baked into the number), and dy/dx_r = 1, so the identity is exact; the
    rank-local parameter gradients then only cover this rank's slice, and the
    explicit all-reduce of the gate gradients in the trainer sums the per-rank
    parts into the exact global gradient (standard DDP argument).
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor) -> torch.Tensor:
        out = t.clone().contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def tp_all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM across TP ranks (no-op when TP is disabled)."""
    if not TP.enabled:
        return t
    return _AllReduceSum.apply(t)


def tp_broadcast(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from `src` (no-op when TP is disabled)."""
    if not TP.enabled:
        return t
    dist.broadcast(t, src=src)
    return t


# ---------------------------------------------------------------------------
# Shard specs
# ---------------------------------------------------------------------------


@dataclass
class ShardSpec:
    """Maps a rank-local Linear/Embedding slice back to its FULL tensor.

    Exactly one of (row_offset/row_len) [column-parallel], row_index [grouped
    column-parallel, e.g. SSM in_proj_qkv], or (col_offset/col_len)
    [row-parallel] is set.  ``vocab_start`` marks a vocabulary-sharded tensor
    (embedding rows / lm_head output columns) so loss code can offset target
    ids.
    """

    full_out: int
    full_in: int
    row_offset: int | None = None
    row_len: int | None = None
    row_index: torch.Tensor | None = None
    col_offset: int | None = None
    col_len: int | None = None
    vocab_start: int | None = None

    @property
    def local_out(self) -> int:
        if self.row_index is not None:
            return int(self.row_index.numel())
        if self.row_len is not None:
            return self.row_len
        return self.full_out

    @property
    def local_in(self) -> int:
        if self.col_len is not None:
            return self.col_len
        return self.full_in

    def describe(self) -> str:
        kind = "row-grouped" if self.row_index is not None else (
            "column" if self.row_len is not None else ("row" if self.col_len is not None else "none")
        )
        return (
            f"{kind}-parallel local [{self.local_out}x{self.local_in}] "
            f"of full [{self.full_out}x{self.full_in}]"
        )

    def slice_weight(self, w_full: torch.Tensor) -> torch.Tensor:
        """Slice a FULL dequantized weight [full_out, padded_full_in] to local."""
        if self.row_index is not None:
            w = w_full[self.row_index.to(w_full.device)]
        elif self.row_offset is not None:
            w = w_full[self.row_offset : self.row_offset + self.row_len]
        else:
            w = w_full
        if self.col_offset is not None:
            w = w[:, self.col_offset : self.col_offset + self.col_len]
        return w

    def slice_qbytes(self, qb_full: torch.Tensor, quant_type: str, padded_full_in: int) -> torch.Tensor:
        """Slice FULL quantized bytes [nblocks, block_bytes] to the local subset.

        Every ggml block format stores blocks row-locally (all blocks of one
        output row are contiguous), so output-row slicing is exact.  Input-column
        slicing is exact only when the column range is block-aligned (all TP
        shard boundaries in this repo's models are multiples of 256).
        Contiguous row slices stay zero-copy views (the candidate cache is
        mmap'd); grouped rows and column slices copy (rank-local size only).
        """
        from voodoo_quant.ggml import get_quant_info

        info = get_quant_info(quant_type)
        bs, bb = info.block_size, info.block_bytes
        if padded_full_in % bs != 0:
            raise ValueError(f"{quant_type}: padded in {padded_full_in} not block-aligned ({bs})")
        nbpr = padded_full_in // bs
        if qb_full.numel() != self.full_out * nbpr * bb:
            raise ValueError(
                f"{quant_type}: qbytes numel {qb_full.numel()} != "
                f"{self.full_out}x{nbpr}x{bb} expected for the FULL tensor"
            )
        q2 = qb_full.reshape(self.full_out, nbpr, bb)
        if self.row_index is not None:
            idx = self.row_index.to(q2.device)
            q2 = q2[idx]
        elif self.row_offset is not None:
            q2 = q2[self.row_offset : self.row_offset + self.row_len]
        if self.col_offset is not None:
            if self.col_offset % bs != 0 or self.col_len % bs != 0:
                raise ValueError(
                    f"{quant_type}: column slice [{self.col_offset}:{self.col_offset + self.col_len}] "
                    f"is not aligned to the {bs}-weight block; TP shard boundaries must be "
                    f"block-aligned for this quant type"
                )
            q2 = q2[:, self.col_offset // bs : (self.col_offset + self.col_len) // bs]
        # .clone() detaches the slice from the FULL staged tensor — a view would
        # pin the full multi-GB buffer alive for the whole run.
        # Row-only slices (and block-aligned column slices) of the mmap'd
        # candidate cache are VIEWS — uploading the view straight to the GPU
        # never materializes a host anon copy (the old .clone() here leaked
        # ~10 GB/rank of glibc-unreturnable fragments across 401 tensors).
        # Only a non-contiguous gather (row_index) needs a copy.
        if self.row_index is not None:
            return q2.reshape(-1, bb).clone()
        return q2.reshape(-1, bb)

    def local_padded_in(self, padded_full_in: int, candidate_types: list[str]) -> int:
        """Padded in_features of the rank-local view (for lazy dequant math)."""
        if self.col_len is None:
            return padded_full_in
        from voodoo_quant.ggml import get_quant_info

        for qt in candidate_types:
            bs = get_quant_info(qt).block_size
            if self.col_len % bs != 0:
                raise ValueError(
                    f"{qt} needs input columns in multiples of {bs}; the rank-local slice is "
                    f"{self.col_len}. Use a TP degree whose shards are block-aligned or drop "
                    f"this candidate type."
                )
        return self.col_len


# ---------------------------------------------------------------------------
# Parallel student layers (nn.Linear / nn.Embedding subclasses so the MixedQuant
# walkers and .to(device) keep working; tp_full_weight is a PLAIN attribute so
# module.to() never moves the mmap reference).
# ---------------------------------------------------------------------------


class ColumnParallelLinear(nn.Linear):
    """Output rows sharded; no communication in forward.

    The weight parameter is the (contiguous) slice of the full mmap'd weight —
    a zero-copy view for row ranges, so sharding the 27B bf16 base costs ~no
    extra host RAM.  Constructed on the meta device first; the parameter is
    then assigned from the slice.
    """

    def __init__(self, weight_slice: torch.Tensor, spec: ShardSpec, full_weight: torch.Tensor,
                 vocab_start: int | None = None):
        super().__init__(spec.full_in, spec.local_out, bias=False,
                         device="meta", dtype=weight_slice.dtype)
        # Keep the slice as a VIEW (row slices of the mmap'd full weight are
        # zero-copy; column slices stay strided — F.linear handles strided
        # weights).  Materializing contiguous copies here costs ~4.7 GB/rank
        # of host anon across the 401 sharded tensors and OOMs a 61 GB host.
        self.weight = nn.Parameter(weight_slice, requires_grad=False)
        self.tp_shard_spec = spec
        self.tp_full_weight = full_weight
        if vocab_start is not None:
            self.tp_vocab_start = vocab_start
            self.tp_vocab_end = vocab_start + spec.local_out


class RowParallelLinear(nn.Linear):
    """Input columns sharded; forward all-reduces the partial sums.

    A column slice of the full weight is NOT contiguous, so it is materialized
    once at rank-local size (e.g. 5120x4352 bf16 ~ 44 MB per MLP down_proj).
    """

    def __init__(self, weight_slice: torch.Tensor, spec: ShardSpec, full_weight: torch.Tensor):
        super().__init__(spec.local_in, spec.full_out, bias=False,
                         device="meta", dtype=weight_slice.dtype)
        self.weight = nn.Parameter(weight_slice.contiguous(), requires_grad=False)
        self.tp_shard_spec = spec
        self.tp_full_weight = full_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
        return tp_all_reduce_sum(F.linear(x, self.weight))


class VocabParallelEmbedding(nn.Embedding):
    """Vocabulary rows sharded; ids outside the shard contribute zeros and the
    per-rank lookups are summed (only the owning rank produces a nonzero row)."""

    def __init__(self, weight_slice: torch.Tensor, spec: ShardSpec, full_weight: torch.Tensor):
        super().__init__(spec.local_out, spec.full_in, device="meta", dtype=weight_slice.dtype)
        # See ColumnParallelLinear: keep the (possibly strided) view.
        self.weight = nn.Parameter(weight_slice)
        self.tp_shard_spec = spec
        self.tp_full_weight = full_weight
        self.tp_vocab_start = spec.vocab_start
        self.tp_vocab_end = spec.vocab_start + spec.local_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dev = self.weight.device
        if input.device != dev:
            input = input.to(dev)
        v0, v1 = self.tp_vocab_start, self.tp_vocab_end
        mask = (input < v0) | (input >= v1)
        local_ids = (input - v0).clamp(0, self.num_embeddings - 1)
        out = F.embedding(local_ids, self.weight)
        if mask.any():
            out = out * (~mask).unsqueeze(-1).to(out.dtype)
        return tp_all_reduce_sum(out)


# ---------------------------------------------------------------------------
# Q8_0-teacher parallel modules (qbytes-backed)
# ---------------------------------------------------------------------------


def _is_q8_module(m) -> bool:
    return (
        isinstance(m, nn.Module)
        and getattr(m, "qweight", None) is not None
        and hasattr(m, "out_features")
    )


def _slice_q8_bytes(qb: torch.Tensor, full_out: int, full_in: int,
                    row_index=None, row_range=None, col_range=None) -> torch.Tensor:
    """Slice Q8_0 qbytes ([nblocks, 34], 32 weights/block) to a rank-local subset."""
    bs, bb = 32, 34
    nbpr = full_in // bs
    q2 = qb.reshape(full_out, nbpr, bb)
    if row_index is not None:
        q2 = q2[row_index.to(q2.device)]
    elif row_range is not None:
        q2 = q2[row_range[0] : row_range[1]]
    if col_range is not None:
        c0, c1 = col_range
        if c0 % bs or (c1 - c0) % bs:
            raise ValueError(f"Q8_0 column slice [{c0}:{c1}] not 32-aligned")
        q2 = q2[:, c0 // bs : c1 // bs]
    return q2.reshape(-1, bb).contiguous()


class RowParallelQ8Linear(nn.Module):
    """Row-parallel wrapper around a Q8Linear holding the local input columns."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.out_features = inner.out_features
        self.in_features = inner.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.inner.qweight.device:
            x = x.to(self.inner.qweight.device)
        return tp_all_reduce_sum(self.inner(x))


class VocabParallelQ8Embedding(nn.Module):
    """Vocabulary-row sharded Q8Embedding (teacher)."""

    def __init__(self, inner: nn.Module, vocab_start: int, local_rows: int):
        super().__init__()
        self.inner = inner
        self.vocab_start = vocab_start
        self.local_rows = local_rows

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dev = self.inner.qweight.device
        if input.device != dev:
            input = input.to(dev)
        mask = (input < self.vocab_start) | (input >= self.vocab_start + self.local_rows)
        local_ids = (input - self.vocab_start).clamp(0, self.local_rows - 1)
        out = self.inner(local_ids)
        if mask.any():
            out = out * (~mask).unsqueeze(-1).to(out.dtype)
        return tp_all_reduce_sum(out)


# ---------------------------------------------------------------------------
# Generic child sharding (works on nn.Linear students and Q8 teacher modules)
# ---------------------------------------------------------------------------


def _shard_child(parent: nn.Module, name: str, *, row_offset=None, row_len=None,
                 row_index=None, col_offset=None, col_len=None, vocab_start=None) -> nn.Module:
    """Replace parent.<name> with a rank-local parallel module in place.

    Idempotent: already-parallel children are returned untouched.
    """
    child = getattr(parent, name)
    if isinstance(child, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding,
                          RowParallelQ8Linear, VocabParallelQ8Embedding)):
        return child

    if isinstance(child, nn.Linear):
        w = child.weight.data
        full_out, full_in = w.shape
        spec = ShardSpec(
            full_out=full_out, full_in=full_in,
            row_offset=row_offset, row_len=row_len, row_index=row_index,
            col_offset=col_offset, col_len=col_len, vocab_start=vocab_start,
        )
        sl = spec.slice_weight(w)
        if col_offset is None:
            new = ColumnParallelLinear(sl, spec, w, vocab_start=vocab_start)
        else:
            new = RowParallelLinear(sl, spec, w)
        setattr(parent, name, new)
        return new

    if _is_q8_module(child):
        qb = child.qweight
        full_out, full_in = child.out_features, child.in_features
        # Shard recipe for per-step reloads: the mmap-backed full qbytes plus
        # this rank's slice parameters (see the trainer's per-step teacher
        # offload/reload cycle).
        _src = getattr(child, "_voodoo_qb_src", None)
        if _src is not None:
            _shard_args = dict(row_index=row_index,
                               row_range=(row_offset, row_offset + row_len) if row_offset is not None else None,
                               col_range=(col_offset, col_offset + col_len) if col_offset is not None else None)
        else:
            _shard_args = None
        if col_offset is not None:
            qb_local = _slice_q8_bytes(qb, full_out, full_in, col_range=(col_offset, col_offset + col_len))
            from voodoo_quant.training.q8_teacher import Q8Linear

            inner = Q8Linear(qb_local, full_out, col_len,
                             None, device=child.qweight.device)
            inner.bias = getattr(child, "bias", None)
            if _shard_args is not None:
                inner._voodoo_qb_shard = (_src[0], full_out, full_in, _shard_args)
            new = RowParallelQ8Linear(inner)
        else:
            if row_index is not None:
                qb_local = _slice_q8_bytes(qb, full_out, full_in, row_index=row_index)
                n_local = int(row_index.numel())
            else:
                qb_local = _slice_q8_bytes(qb, full_out, full_in,
                                           row_range=(row_offset, row_offset + row_len))
                n_local = row_len
            from voodoo_quant.training.q8_teacher import Q8Linear

            new = Q8Linear(qb_local, n_local, full_in,
                           getattr(child, "bias", None), device=child.qweight.device)
            if _shard_args is not None:
                new._voodoo_qb_shard = (_src[0], full_out, full_in, _shard_args)
            if vocab_start is not None:
                new.tp_vocab_start = vocab_start
                new.tp_vocab_end = vocab_start + n_local
        setattr(parent, name, new)
        return new

    if isinstance(child, nn.Embedding) or (
        isinstance(child, nn.Module) and getattr(child, "qweight", None) is not None
        and hasattr(child, "num_embeddings")
    ):
        tp = TP.world_size
        if getattr(child, "qweight", None) is not None:
            # Teacher Q8Embedding: slice qbytes rows, wrap with vocab masking.
            full_vocab = child.num_embeddings
            emb_dim = child.embedding_dim
            assert vocab_start is not None, "embedding sharding requires vocab_start"
            assert full_vocab % tp == 0, f"vocab {full_vocab} not divisible by tp={tp}"
            lv = full_vocab // tp
            qb_local = _slice_q8_bytes(child.qweight, full_vocab, emb_dim,
                                       row_range=(vocab_start, vocab_start + lv))
            from voodoo_quant.training.q8_teacher import Q8Embedding

            inner = Q8Embedding(qb_local, lv, emb_dim, device=child.qweight.device)
            _src = getattr(child, "_voodoo_qb_src", None)
            if _src is not None:
                inner._voodoo_qb_shard = (_src[0], full_vocab, emb_dim,
                                           dict(row_range=(vocab_start, vocab_start + lv)))
            new = VocabParallelQ8Embedding(inner, vocab_start, lv)
            setattr(parent, name, new)
            return new
        w = child.weight.data
        full_vocab, emb_dim = w.shape
        spec = ShardSpec(full_out=full_vocab, full_in=emb_dim,
                         row_offset=vocab_start, row_len=full_vocab // TP.world_size,
                         vocab_start=vocab_start)
        new = VocabParallelEmbedding(spec.slice_weight(w), spec, w)
        setattr(parent, name, new)
        return new

    raise TypeError(f"parallel: cannot shard {type(child).__name__} at '{name}'")


# ---------------------------------------------------------------------------
# Per-submodule TP adapters
# ---------------------------------------------------------------------------


def tp_attention(attn: nn.Module):
    """Shard a Qwen3_5Attention by heads.

    q_proj's output is [heads, head_dim*2] (per-head q|gate pairs), so a
    contiguous per-rank head range keeps each head's q and gate together; the
    stock forward's view/chunk(2) then works unchanged on the local slice.
    """
    tp, r = TP.world_size, TP.rank
    q = attn.q_proj
    assert q.out_features % tp == 0, f"q_proj out {q.out_features} not divisible by tp={tp}"
    kv = getattr(attn, "k_proj")
    head_dim = getattr(attn, "head_dim", 0) or 1
    assert kv.out_features % (tp * head_dim) == 0, (
        f"k_proj out {kv.out_features} cannot be split into whole heads across tp={tp} "
        f"(head_dim {head_dim}); TP degree must divide the KV-head count"
    )
    lq = q.out_features // tp
    _shard_child(attn, "q_proj", row_offset=r * lq, row_len=lq)
    for nm in ("k_proj", "v_proj"):
        m = getattr(attn, nm)
        assert m.out_features % tp == 0, f"{nm} out {m.out_features} not divisible by tp={tp}"
        lk = m.out_features // tp
        _shard_child(attn, nm, row_offset=r * lk, row_len=lk)
    o = attn.o_proj
    assert o.in_features % tp == 0, f"o_proj in {o.in_features} not divisible by tp={tp}"
    lo = o.in_features // tp
    _shard_child(attn, "o_proj", col_offset=r * lo, col_len=lo)
    head_dim = getattr(attn, "head_dim", 0) or 1
    attn.tp_heads_local = q.out_features // (2 * head_dim * tp)


def tp_ssm(gdn: nn.Module):
    """Shard a Qwen3_5GatedDeltaNet by v-heads (k-heads shard in lockstep).

    in_proj_qkv's output layout is [q(key_dim) | k(key_dim) | v(value_dim)];
    the rank-local slice concatenates this rank's q, k and v head ranges, which
    matches both the conv1d channel order and the torch.split sizes after the
    module's key_dim/value_dim attributes are made rank-local.  The stock
    forward then runs unchanged.
    """
    tp, r = TP.world_size, TP.rank
    kd, vd = gdn.key_dim, gdn.value_dim
    assert kd % tp == 0 and vd % tp == 0, f"SSM key/value dims ({kd}/{vd}) not divisible by tp={tp}"
    kl, vl = kd // tp, vd // tp
    rows = torch.cat([
        torch.arange(r * kl, (r + 1) * kl),
        torch.arange(kd + r * kl, kd + (r + 1) * kl),
        torch.arange(2 * kd + r * vl, 2 * kd + (r + 1) * vl),
    ])
    _shard_child(gdn, "in_proj_qkv", row_index=rows)
    _shard_child(gdn, "in_proj_z", row_offset=r * vl, row_len=vl)
    nv = gdn.num_v_heads
    assert nv % tp == 0, f"num_v_heads {nv} not divisible by tp={tp}"
    nvl = nv // tp
    _shard_child(gdn, "in_proj_b", row_offset=r * nvl, row_len=nvl)
    _shard_child(gdn, "in_proj_a", row_offset=r * nvl, row_len=nvl)
    _shard_child(gdn, "out_proj", col_offset=r * vl, col_len=vl)

    # Depthwise conv: channels follow the same [q|k|v] order as in_proj_qkv.
    conv = gdn.conv1d
    k = conv.kernel_size
    local_dim = 2 * kl + vl
    new_conv = nn.Conv1d(local_dim, local_dim, kernel_size=k, groups=local_dim,
                         padding=conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding,
                         bias=conv.bias is not None)
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.data[rows.to(conv.weight.device)])
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data[rows.to(conv.bias.device)])
    new_conv = new_conv.to(conv.weight.device, conv.weight.dtype)
    gdn.conv1d = new_conv

    # Per-v-head scalars and rank-local dimension bookkeeping.
    with torch.no_grad():
        gdn.dt_bias.data = gdn.dt_bias.data[r * nvl : (r + 1) * nvl].clone()
        gdn.A_log.data = gdn.A_log.data[r * nvl : (r + 1) * nvl].clone()
    gdn.num_v_heads = nvl
    gdn.num_k_heads = gdn.num_k_heads // tp
    gdn.key_dim = kl
    gdn.value_dim = vl
    gdn.conv_dim = local_dim


def tp_mlp(mlp: nn.Module):
    tp, r = TP.world_size, TP.rank
    g = mlp.gate_proj
    assert g.out_features % tp == 0, f"mlp intermediate {g.out_features} not divisible by tp={tp}"
    li = g.out_features // tp
    _shard_child(mlp, "gate_proj", row_offset=r * li, row_len=li)
    _shard_child(mlp, "up_proj", row_offset=r * li, row_len=li)
    _shard_child(mlp, "down_proj", col_offset=r * li, col_len=li)


def _shard_lm_head(model: nn.Module):
    """Column-parallel lm_head over the vocabulary (per-rank logits, no comm)."""
    head = None
    holder = None
    for cand_holder, attr in ((model, "lm_head"), (getattr(model, "model", None), "lm_head")):
        if cand_holder is not None and getattr(cand_holder, attr, None) is not None:
            head, holder = getattr(cand_holder, attr), cand_holder
            break
    if head is None:
        return
    print(f"  [lmhead-shard] head type={type(head).__name__} out={getattr(head,'out_features','?')}", flush=True)
    if isinstance(head, (ColumnParallelLinear, RowParallelQ8Linear)):
        return
    tp, r = TP.world_size, TP.rank
    full_vocab = head.out_features
    assert full_vocab % tp == 0, f"vocab {full_vocab} not divisible by tp={tp}"
    lv = full_vocab // tp
    _shard_child(holder, "lm_head", row_offset=r * lv, row_len=lv, vocab_start=r * lv)


def tp_patch_model(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Shard a model in place for the current TP rank via its arch adapter.

    Mechanism lives here (``_shard_child`` and friends); policy lives in
    :mod:`voodoo_quant.arch` adapters, resolved from the model's config. A
    model with no registered adapter is REFUSED — sharding blindly would
    silently corrupt all-reduce sums. Weights must be materialized (CPU or
    GPU) but will typically be mmap'd checkpoint pages shared across ranks.
    """
    from voodoo_quant.arch import adapter_for_model

    adapter = adapter_for_model(model)
    if adapter is None:
        raise RuntimeError(
            f"no architecture adapter registered for {type(model).__name__} "
            f"(model_type={getattr(getattr(model, 'config', None), 'model_type', '?')}); "
            "tensor parallelism is refused rather than sharded blindly. "
            "Add an adapter in voodoo_quant/arch/ or run single-GPU / --device_map."
        )
    text = getattr(model, "model", None)
    if text is None or not hasattr(text, "layers"):
        text = model
    embed = getattr(text, "embed_tokens", None)
    n_handled = 0
    if embed is not None:
        tp, r = TP.world_size, TP.rank
        # Q8Embedding (teacher) exposes num_embeddings too, without .weight.
        full_vocab = getattr(embed, "num_embeddings", None)
        if full_vocab is None:
            full_vocab = embed.weight.shape[0]
        assert full_vocab % tp == 0, f"vocab {full_vocab} not divisible by tp={tp}"
        _shard_child(text, "embed_tokens", row_offset=r * (full_vocab // tp),
                     row_len=full_vocab // tp, vocab_start=r * (full_vocab // tp))
    for layer in text.layers:
        n_handled += len(adapter.shard_layer(layer))
        # Upload this layer's rank-local shard immediately: the sharded views/
        # slices then never sit in host anon (row slices are mmap views, but
        # holding all 64 layers' shards on CPU costs ~13.5 GB/rank and, with 4
        # ranks + init transients, OOM-kills the host before training starts).
        _dev = TP.device
        if _dev is not None and str(_dev).startswith("cuda"):
            try:
                layer.to(_dev)
            except Exception:
                pass
    _shard_lm_head(model)
    if verbose:
        print(
            f"[tp] patched rank {TP.rank} ({adapter.__name__}): {n_handled} layer "
            f"submodule groups, vocab-parallel embed + lm_head",
            flush=True,
        )
    return model


def move_tp_model_to_device(model: nn.Module, device: torch.device):
    """Move a TP-patched model to its rank device (tp_full_weight refs stay on CPU)."""
    model.to(device)


# ---------------------------------------------------------------------------
# Smoke test (CPU-safe: no flash-attn / ROCm dependency — it only exercises
# the collectives autograd Function and the parallel layer math).
#
#   torchrun --standalone --nproc_per_node=4 -m voodoo_quant.parallel
#   python -m voodoo_quant.parallel   # single-process degenerate run
#
# On 4x MI100 (gfx908) hosts NCCL_P2P_DISABLE=1 has been required: with P2P
# enabled, 4-rank RCCL collectives triggered GPU page faults and the ranks
# were SIGKILLed.
# ---------------------------------------------------------------------------


def _smoke_test() -> int:
    rank, world = init_tp()
    device = TP.device if TP.enabled else (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )

    def log(msg: str) -> None:
        if rank == 0:
            print(f"[tp-smoke rank {rank}/{world}] {msg}", flush=True)

    log(f"initialized on {device} (backend={dist.get_backend() if TP.enabled else 'none'})")

    # Identical full weights on every rank (fixed seed).
    torch.manual_seed(1234)
    batch, seq = 2, 8
    in_dim, mid, out_dim = 256, 512, 128
    vocab = 256  # divisible by any tp degree tested here
    failures = 0

    def check(name: str, got: torch.Tensor, ref: torch.Tensor,
              rtol: float, atol: float) -> None:
        nonlocal failures
        err = (got.float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(got.float(), ref.float(), rtol=rtol, atol=atol)
            log(f"  {name}: PASS (max abs err {err:.3e})")
        except AssertionError as e:
            failures += 1
            log(f"  {name}: FAIL (max abs err {err:.3e})\n{e}")

    for dtype, rtol, atol in (
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 3e-2, 3e-2),
    ):
        tag = str(dtype).split(".")[-1]
        w1 = torch.randn(mid, in_dim, dtype=dtype, device=device) * 0.1
        w2 = torch.randn(out_dim, mid, dtype=dtype, device=device) * 0.1
        x = torch.randn(batch, seq, in_dim, dtype=dtype, device=device)
        g = torch.randn(batch, seq, out_dim, dtype=dtype, device=device)

        # Unsharded reference with input gradients.
        x_ref = x.clone().requires_grad_(True)
        h_ref = torch.nn.functional.gelu(torch.nn.functional.linear(x_ref, w1))
        y_ref = torch.nn.functional.linear(h_ref, w2)
        (y_ref * g).sum().backward()

        # Column(gate/up) -> gelu -> Row(down): the standard TP MLP pattern.
        col_spec = ShardSpec(
            full_out=mid, full_in=in_dim,
            row_offset=TP.rank * (mid // world), row_len=mid // world,
        )
        row_spec = ShardSpec(
            full_out=out_dim, full_in=mid,
            col_offset=TP.rank * (mid // world), col_len=mid // world,
        )
        col = ColumnParallelLinear(col_spec.slice_weight(w1), col_spec, w1).to(device)
        row = RowParallelLinear(row_spec.slice_weight(w2), row_spec, w2).to(device)

        x_tp = x.clone().requires_grad_(True)
        y_tp = row(torch.nn.functional.gelu(col(x_tp)))
        check(f"col+row fwd bf16={tag}", y_tp, y_ref.detach(), rtol, atol)

        # Rank-local input grads are partial; their SUM across TP ranks must
        # equal the unsharded grad (the DDP-style argument in _AllReduceSum).
        (y_tp * g).sum().backward()
        grad_sum = all_reduce_sum(x_tp.grad.clone())
        check(f"col+row dL/dx summed bf16={tag}", grad_sum, x_ref.grad, rtol, atol)

        # Vocabulary-parallel embedding vs the full lookup.
        emb_w = torch.randn(vocab, in_dim, dtype=dtype, device=device) * 0.1
        ids = torch.randint(0, vocab, (batch, seq), device=device)
        emb_spec = ShardSpec(
            full_out=vocab, full_in=in_dim,
            row_offset=TP.rank * (vocab // world), row_len=vocab // world,
            vocab_start=TP.rank * (vocab // world),
        )
        emb = VocabParallelEmbedding(emb_spec.slice_weight(emb_w), emb_spec, emb_w).to(device)
        check(f"vocab embed bf16={tag}", emb(ids),
              torch.nn.functional.embedding(ids, emb_w), rtol, atol)

    # Aggregate pass/fail across ranks, then tear down.
    fail_t = all_reduce_sum(torch.tensor([float(failures)], device=device))
    total = int(fail_t.item())
    barrier()
    log("SMOKE TEST PASSED" if total == 0 else f"SMOKE TEST FAILED: {total} failure(s)")
    tp_finalize()
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_smoke_test())

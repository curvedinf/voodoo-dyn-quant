"""GPU-resident Q8_0 teacher for Voodoo gate training.

The teacher checkpoint (``qwen38_27b_lm_teacher_q8_0.pt``, built by the
teacher-creation script) stores every Linear/Embedding weight as llama.cpp Q8_0
bytes plus ``quant_meta`` describing shapes.  Loading the teacher as BF16 would
cost ~54 GB; as Q8_0 bytes it is ~27 GB, and each shard dequantizes once at init
into a bf16 Linear/Embedding that runs at normal speed during the (no-grad)
teacher forward.

The module layout mirrors whatever HF model instance it wraps: ``wrap(model)``
replaces every ``nn.Linear``/``nn.Embedding`` whose ``<name>.weight`` is a
Q8_0 tensor in the checkpoint, leaving everything else untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from voodoo_quant.ggml import dequantize_tensor

# Imported lazily to avoid a circular import at module load.
def _gpu_dequant(qb, out_features, in_features, device):
    try:
        from voodoo_quant.layers import dequantize_tensor_gpu

        return dequantize_tensor_gpu(qb, "Q8_0", out_features, in_features, device)
    except torch.OutOfMemoryError:
        raise
    except Exception:
        w = dequantize_tensor(qb.cpu(), "Q8_0", out_features, in_features)
        return w.to(device=device, dtype=torch.bfloat16)


_BIG_MODULE_ELEMS = 64_000_000  # >256 MiB bf16: dequantize in row blocks


def _dequant_blocked(qb, out_features, in_features, device, dtype=torch.bfloat16):
    """Row-blocked Q8_0 dequant for vocabulary-scale tensors.

    The single-shot path materializes an fp32 intermediate ~2x the bf16 weight
    (4.7 GiB for 248320x5120) plus the bf16 result — an OOM on a loaded shard.
    Blocks cost the same flops with ~256 MiB transients.
    """
    from voodoo_quant.layers import dequantize_tensor_gpu, _quant_info_for

    info = _quant_info_for("Q8_0")
    nbpr = in_features // info.block_size
    out = torch.empty(out_features, in_features, device=device, dtype=dtype)
    block_rows = max(1, 8_000_000 // in_features)
    qb2d = qb.view(out_features, nbpr, info.block_bytes)
    for r0 in range(0, out_features, block_rows):
        r1 = min(out_features, r0 + block_rows)
        sub = qb2d[r0:r1].reshape((r1 - r0) * nbpr, info.block_bytes)
        blk = dequantize_tensor_gpu(sub, "Q8_0", r1 - r0, in_features, device)
        out[r0:r1] = blk[:, :in_features].to(dtype)
        del blk
    return out


class Q8Linear(nn.Module):
    """nn.Linear stand-in keeping Q8_0 bytes on-GPU, dequantized per forward.

    Dequantizing once at load time costs full BF16 memory (teacher == student
    footprint, which OOMs 4x32 GB); keeping the ~8.5-bit bytes and dequantizing
    in forward (no-grad, GPU kernels verified bit-identical) matches the recipe
    budget of ~6.5 GiB/GPU for the teacher.
    """

    def __init__(self, qbytes: torch.Tensor, out_features: int, in_features: int,
                 bias: torch.Tensor | None = None, device: torch.device | None = None):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        dev = torch.device(device) if device is not None else qbytes.device
        qb = qbytes.to(dev) if qbytes.device != dev else qbytes
        # Registered as a buffer so `_shard_model`'s `.to()` relocates it.
        self.register_buffer("qweight", qb)
        self._w: torch.Tensor | None = None  # dequant cache (device-checked)
        self._weight_is_big = out_features * in_features > _BIG_MODULE_ELEMS
        if bias is not None:
            self.bias = nn.Parameter(bias.to(dev).to(torch.bfloat16), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def _weight(self) -> torch.Tensor:
        if self.qweight.numel() == 0:
            raise RuntimeError(f"Q8Linear empty qweight: {self.out_features}x{self.in_features} (offloaded, never reloaded)")
        if self._w is None or self._w.device != self.qweight.device:
            if self.out_features * self.in_features > _BIG_MODULE_ELEMS:
                self._w = _dequant_blocked(
                    self.qweight, self.out_features, self.in_features, self.qweight.device
                )
            else:
                self._w = _gpu_dequant(
                    self.qweight, self.out_features, self.in_features, self.qweight.device
                )
        return self._w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_is_big and x.shape[0] * self.in_features <= 16_000_000:
            # Vocab-blocked matmul for the teacher lm_head: the full bf16
            # weight (2.37 GiB) never materializes; logits are computed in
            # ~0.3 GiB output-row blocks.  Exact (blockwise matmul slices).
            from voodoo_quant.layers import dequantize_tensor_gpu, _quant_info_for

            dev = self.qweight.device
            info = _quant_info_for("Q8_0")
            nbpr = self.in_features // info.block_size
            qb2d = self.qweight.view(self.out_features, nbpr, info.block_bytes)
            block_rows = max(1, 8_000_000 // self.in_features)
            outs = []
            for r0 in range(0, self.out_features, block_rows):
                r1 = min(self.out_features, r0 + block_rows)
                sub = qb2d[r0:r1].reshape((r1 - r0) * nbpr, info.block_bytes)
                blk = dequantize_tensor_gpu(sub, "Q8_0", r1 - r0, self.in_features, dev)
                outs.append(F.linear(x, blk[:, : self.in_features].to(x.dtype)))
                del blk
            return torch.cat(outs, dim=-1)
        w = self._weight().to(x.dtype)
        out = F.linear(x, w, self.bias if self.bias is not None else None)
        # One-shot cache: teacher modules run inside torch.no_grad, once per
        # step.  Keeping every dequantized weight pinned doubles teacher VRAM
        # (~54 GB across GPUs vs the ~26 GB Q8 budget).  Free after each use;
        # next forward dequantizes again (~ms with the GPU kernels).
        self._w = None
        return out


class Q8Embedding(nn.Module):
    """nn.Embedding stand-in keeping Q8_0 bytes on-GPU, dequantized per forward."""

    def __init__(self, qbytes: torch.Tensor, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        dev = torch.device(device) if device is not None else qbytes.device
        qb = qbytes.to(dev) if qbytes.device != dev else qbytes
        self.register_buffer("qweight", qb)
        self._w: torch.Tensor | None = None

    def _weight(self) -> torch.Tensor:
        if self._w is None or self._w.device != self.qweight.device:
            if self.num_embeddings * self.embedding_dim > _BIG_MODULE_ELEMS:
                self._w = _dequant_blocked(
                    self.qweight, self.num_embeddings, self.embedding_dim, self.qweight.device
                )
            else:
                self._w = _gpu_dequant(
                    self.qweight, self.num_embeddings, self.embedding_dim, self.qweight.device
                )
        return self._w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Row-select: an embedding forward only touches the batch's rows
        # (<= seq_len x batch), so dequantize just those instead of the full
        # vocabulary matrix (2.37 GiB bf16 for 248320x5120 — an OOM on a
        # loaded shard).  Q8_0 blocks are row-local, so slicing qbytes rows is
        # exact.
        from voodoo_quant.layers import dequantize_tensor_gpu, _quant_info_for

        dev = self.qweight.device
        rows, inverse = torch.unique(x.reshape(-1), return_inverse=True)
        rows = rows.to(dev)
        info = _quant_info_for("Q8_0")
        nbpr = self.embedding_dim // info.block_size
        qb2d = self.qweight.view(self.num_embeddings, nbpr, info.block_bytes)
        sub = qb2d[rows].reshape(rows.numel() * nbpr, info.block_bytes)
        w_rows = dequantize_tensor_gpu(sub, "Q8_0", rows.numel(), self.embedding_dim, dev)
        w_rows = w_rows.to(torch.bfloat16)
        return w_rows[inverse].reshape(*x.shape, self.embedding_dim)


def wrap_q8_teacher(model: nn.Module, teacher_state_dict: dict, quant_meta: dict,
                    layer_devices: dict[int, torch.device] | None = None) -> nn.Module:
    """Replace teacher Linears/Embeddings with Q8-dequantized modules in place.

    Args:
        model: a freshly built (empty) model instance of the right architecture.
        teacher_state_dict: the Q8_0 teacher checkpoint's ``model_state_dict``.
        quant_meta: per-tensor metadata (out_features/in_features) from the same
            checkpoint; tensors absent from it are carried over as-is.
        layer_devices: optional {layer_idx: device} used to build the teacher
            with the same layer-wise sharding as the student.

    The checkpoint uses stripped names (``layers.N...``) while the HF model uses
    ``model.layers.N...``; lookups try the module-relative name first and the
    stripped key as a fallback.
    """
    if layer_devices:
        # Place layer modules first so wrapped weights are dequantized straight
        # onto their shard device.
        for name, module in model.named_modules():
            idx = None
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "layers":
                try:
                    idx = int(parts[-1])
                except ValueError:
                    idx = None
            dev = layer_devices.get(idx) if idx is not None else None
            if dev is not None and name.count(".") <= 2:
                module._q8_target_device = dev

    def _lookup(name: str):
        for key in (name, name.removeprefix("model.")):
            if key in teacher_state_dict:
                return teacher_state_dict[key]
        return None

    replaced = 0
    wrapped_keys: set[str] = set()
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            weight_key = full + ".weight"
            qb = _lookup(weight_key)
            if qb is None or qb.dim() != 2 or qb.dtype != torch.uint8:
                continue
            meta = (
                quant_meta.get(weight_key)
                or quant_meta.get(weight_key.removeprefix("model."))
                or quant_meta.get(full)
                or quant_meta.get(full.removeprefix("model."))
            )
            if meta is None:
                print(f"  q8 teacher WARNING: no quant_meta for {full}; skipping", flush=True)
                continue
            if isinstance(child, nn.Linear):
                new = Q8Linear(qb, meta["out_features"], meta["in_features"],
                               child.bias.detach() if child.bias is not None else None,
                               device=getattr(module, "_q8_target_device", None))
            elif isinstance(child, nn.Embedding):
                new = Q8Embedding(qb, child.num_embeddings, child.embedding_dim,
                                  device=getattr(module, "_q8_target_device", None))
            else:
                continue
            # Remember the mmap-backed source (file-backed pages, no anon cost)
            # so a TP run can drop and re-create the GPU buffer per step
            # (see the trainer's per-step teacher offload/reload).
            new._voodoo_qb_src = (qb,)
            # Default shard descriptor = the FULL tensor; TP sharding overwrites
            # this for sharded modules (see voodoo_quant.parallel._shard_child).
            # Ensures the per-step offload/reload cycle can restore every Q8
            # module, sharded or not.
            new._voodoo_qb_shard = (qb, meta["out_features"], meta["in_features"], {})
            setattr(module, child_name, new)
            replaced += 1
            wrapped_keys.add(weight_key)
            wrapped_keys.add(weight_key.removeprefix("model."))

    # Carry over every non-Q8 tensor (norms, biases, conv1d, A_log, dt_bias...).
    # Wrapped keys are excluded: their Q8 bytes are consumed above, and loading
    # them again would collide with the dequantized bf16 params.  The teacher
    # model nests its text stack as `model.`, the checkpoint stores stripped
    # `layers.N...` keys; remap so the assign-load actually hits.
    def _carry_key(k: str) -> str:
        if k.startswith("model.") or k == "lm_head.weight":
            return k
        return "model." + k

    carry_sd = {_carry_key(k): v for k, v in teacher_state_dict.items() if _carry_key(k) not in wrapped_keys and k not in wrapped_keys}
    missing, unexpected = model.load_state_dict(carry_sd, strict=False, assign=True)
    carried = len(carry_sd)
    print(f"  q8 teacher: wrapped {replaced} Q8_0 tensors; carried {carried} as-is "
          f"({len(missing)} missing, {len(unexpected)} unexpected)", flush=True)
    if unexpected:
        print(f"  q8 teacher WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})", flush=True)
    if layer_devices:
        for name, param in model.named_parameters():
            idx = None
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "layers":
                try:
                    idx = int(parts[-1])
                except ValueError:
                    idx = None
            dev = layer_devices.get(idx) if idx is not None else None
            if dev is not None and param.device.type == "cpu":
                param.data = param.data.to(dev)
        for name, buf in model.named_buffers():
            idx = None
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "layers":
                try:
                    idx = int(parts[-1])
                except ValueError:
                    idx = None
            dev = layer_devices.get(idx) if idx is not None else None
            if dev is not None and buf.device.type == "cpu":
                buf.data = buf.data.to(dev)
    return model

"""Sensitivity-based layout optimization: warm-start and knapsack polish.

Both address the same root cause from opposite ends — the gate-training loop
structurally couldn't see where quality lives (the V50 outcome): logit-level
KL is a poor attributor for deep-stack MLP precision, so attention buys look
like cheap wins and MLP buys look like noise.

- **Warm start** (exploration end): before training, measure each tensor's
  real output perturbation per candidate on calibration activations and
  initialize the gates from a greedy knapsack over quality-per-byte, instead
  of a blind zero start.
- **Knapsack polish** (termination end): after the argmax freezes (and after
  ``--tensor_upgrades``), enumerate one-rung moves per tensor, rank by
  measured benefit-per-byte, and greedily apply moves that fit the budget's
  dead zone. This can execute precisely the "swap one rung on one tensor"
  move that dead-temperature gradients cannot.

The expensive ingredient — exact quantize/dequantize of every tensor x
candidate — is already the candidate cache; the new work is one hooked
forward pass to capture input activations X_t plus one matmul per
tensor-candidate pair.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

import torch

from voodoo_quant.ggml import QUANT_LADDER, bytes_per_weight


@dataclass
class SensitivityTable:
    """Per-tensor sensitivity and price over its candidate menu.

    ``sens[t][q]``  = ||(W_q - W) X_t||^2 / ||W X_t||^2   (relative output perturbation)
    ``bytes[t][q]`` = exact stored bytes for tensor t at quant q
    ``price[t][q]`` = sens / bytes  (lower is a better buy: cheap AND faithful)
    """

    sens: dict[str, dict[str, float]] = field(default_factory=dict)
    bytes_: dict[str, dict[str, float]] = field(default_factory=dict)
    input_norms: dict[str, float] = field(default_factory=dict)
    candidates: dict[str, list[str]] = field(default_factory=dict)

    def price(self, tensor: str, quant: str) -> float:
        b = self.bytes_[tensor][quant]
        s = self.sens[tensor][quant]
        return s / b if b > 0 else float("inf")


def _iter_batches(dataloader, device, max_batches: int, max_tokens: int):
    """Yield (x, y) batches, capped by count and total tokens."""
    n = 0
    tokens = 0
    for x, y in dataloader:
        yield x.to(device), y.to(device)
        n += 1
        tokens += int(x.numel())
        if max_batches and n >= max_batches:
            break
        if max_tokens and tokens >= max_tokens:
            break


@torch.no_grad()
def capture_layer_inputs(
    model,
    replaced: dict,
    dataloader,
    device,
    max_batches: int = 4,
    max_tokens: int = 8192,
) -> dict[str, torch.Tensor]:
    """One hooked forward pass per batch: collect the inputs entering every
    selectable module (a representative mean over rows is enough for the
    Gram accumulation below — we keep the raw [tokens, in_features] for a
    few batches, bounded by max_tokens).

    Hooks run on the MixedQuant modules themselves: their input IS X_t.
    """
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in replaced}
    handles = []

    def make_hook(name):
        def hook(mod, args, kwargs):
            x = args[0] if args else kwargs.get("input")
            if isinstance(x, torch.Tensor) and x.dim() >= 2 and x.shape[-1] == mod.in_features:
                flat = x.reshape(-1, x.shape[-1]).detach().to(torch.float32)
                # cap per-tensor accumulation to keep CPU RAM bounded
                if flat.shape[0] > 4096:
                    idx = torch.randperm(flat.shape[0])[:4096].to(flat.device)
                    flat = flat[idx]
                captured[name].append(flat.cpu())
            return None

        return hook

    for name, mod in replaced.items():
        handles.append(mod.register_forward_pre_hook(make_hook(name), with_kwargs=True))

    was_training = model.training
    model.eval()
    try:
        for x, _y in _iter_batches(dataloader, device, max_batches, max_tokens):
            model(input_ids=x)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    out = {}
    for name, chunks in captured.items():
        if chunks:
            t = torch.cat(chunks, dim=0)
            if t.shape[0] > 8192:
                idx = torch.randperm(t.shape[0])[:8192]
                t = t[idx]
            out[name] = t
    gc.collect()
    return out


@torch.no_grad()
def build_sensitivity_table(
    replaced: dict,
    inputs: dict[str, torch.Tensor],
    device_for_math: torch.device | None = None,
    max_rows: int = 4096,
) -> SensitivityTable:
    """Measure s(t, q) for every tensor x candidate from cached activations.

    For each tensor t with source weight W (full, bf16) and cached candidates
    W_q (the pre-dequantized candidate cache — exactly llama.cpp's output):

        s(t, q) = ||(W_q - W) X_t||^2 / ||W X_t||^2

    computed as one matmul per candidate against a row-capped activation
    sample. Fallback when a candidate buffer is missing (lazy mode): the
    source weight is requantized via the exact ggml path.
    """
    from voodoo_quant.ggml import dequantize_tensor, quantize_tensor
    from voodoo_quant.layers import _quant_info_for

    table = SensitivityTable()
    dev = device_for_math or torch.device("cpu")

    for name, mod in replaced.items():
        cands = list(mod.candidate_types)
        table.candidates[name] = cands
        table.sens[name] = {}
        table.bytes_[name] = {}

        X = inputs.get(name)
        if X is None or X.numel() == 0:
            # no activations captured (module untouched by the sample):
            # fall back to weight-space relative error as a crude price
            X = None

        # the source weight: prefer the module's retained full weight;
        # MixedQuant drops .weight, so recover the FINEST candidate by
        # ladder rank (Q8_0 when present, else the top of its menu) as the
        # highest-fidelity reference for the source.
        from voodoo_quant.ggml import QUANT_LADDER as _LADDER
        finest = next((q for q in reversed(_LADDER) if q in cands), None)
        ref = getattr(mod, f"_w_{cands.index(finest)}", None) if finest is not None else None
        if ref is None:
            ref = getattr(mod, "_w_0", None)
        full_w = None if ref is None else ref.to(dev).float()

        n_rows = min(X.shape[0], max_rows) if X is not None else 0
        Xd = X[:n_rows].to(dev) if X is not None else None

        for k, qt in enumerate(cands):
            info = _quant_info_for(qt)
            padded_in = mod.padded_in_features
            numel = mod.out_features * padded_in
            table.bytes_[name][qt] = numel * bytes_per_weight(qt)

            cand = getattr(mod, f"_w_{k}", None)
            if cand is None:  # lazy mode: exact requantize on the spot
                qb = getattr(mod, f"_qweight_{k}", None)
                if qb is None:
                    continue
                cand = dequantize_tensor(qb.cpu(), qt, mod.out_features, padded_in)
            if cand is None:
                continue
            Wq = cand.to(dev).float()

            if Xd is not None and full_w is not None:
                base = (full_w @ Xd.T).pow(2).sum()
                delta = ((Wq - full_w) @ Xd.T).pow(2).sum()
                sens = (delta / base.clamp_min(1e-12)).item()
            else:
                # weight-space fallback: relative Frobenius perturbation
                if full_w is None:
                    continue
                sens = ((Wq - full_w).norm() / full_w.norm().clamp_min(1e-12)).item()
            table.sens[name][qt] = sens
            del Wq

        del Xd, full_w
        if len(table.sens) % 64 == 0:
            gc.collect()

    return table


def greedy_knapsack_layout(
    table: SensitivityTable,
    budget_bytes: float,
    non_targeted_bytes: float,
) -> dict[str, str]:
    """Greedy quality-per-byte packing under the budget.

    Start every tensor at its cheapest candidate; repeatedly upgrade the
    tensor whose next rung-up has the best marginal (sens_now - sens_next)
    per extra byte, while the total fits. This is the classic greedy
    fractional-knapsack heuristic — per-tensor rungs make it a discrete
    staircase, and sens deltas are (empirically) convex in quality, which is
    exactly where greedy does well.
    """
    layout: dict[str, str] = {}
    # ladder rank per tensor from its own candidate menu (menus may differ per role)
    rank = {t: [q for q in QUANT_LADDER if q in table.candidates.get(t, [])] for t in table.sens}
    for t, lad in rank.items():
        if not lad:
            continue
        layout[t] = lad[0]

    def total() -> float:
        return non_targeted_bytes + sum(table.bytes_[t][q] for t, q in layout.items())

    # step 1: mandatory cheapest assignment
    # step 2: greedy upgrades
    while True:
        best = None  # (benefit_per_byte, tensor, next_q)
        for t, q in layout.items():
            lad = rank[t]
            i = lad.index(q)
            if i + 1 >= len(lad):
                continue
            nxt = lad[i + 1]
            s_now, s_next = table.sens[t].get(q), table.sens[t].get(nxt)
            if s_now is None or s_next is None:
                continue
            benefit = max(s_now - s_next, 0.0)
            cost = table.bytes_[t][nxt] - table.bytes_[t][q]
            if cost <= 0 or benefit <= 0:
                continue
            bp = benefit / cost
            if best is None or bp > best[0]:
                best = (bp, t, nxt)
        if best is None:
            break
        _, t, nxt = best
        new_total = total() - table.bytes_[t][layout[t]] + table.bytes_[t][nxt]
        if new_total > budget_bytes:
            # this upgrade does not fit; try excluding and continue with others
            rank[t] = rank[t][: rank[t].index(nxt)]  # pin at current rung
            if all(layout[tt] == rank[tt][-1] for tt in layout):
                break
            continue
        layout[t] = nxt
    return layout


def warm_start_gates(replaced: dict, layout: dict[str, str], logit_bias: float = 4.0) -> int:
    """Seed gate logits so the softmax starts concentrated on `layout`.

    Sets the assigned candidate's logit to +logit_bias and others to 0 (the
    trainer's anneal then sharpens further). Returns the number of tensors
    seeded.
    """
    seeded = 0
    with torch.no_grad():
        for name, mod in replaced.items():
            want = layout.get(name)
            if want is None or want not in mod.candidate_types:
                continue
            g = torch.zeros_like(mod.gates)
            g[mod.candidate_types.index(want)] = logit_bias
            mod.gates.copy_(g)
            seeded += 1
    return seeded


def knapsack_polish(
    layout: dict[str, str],
    table: SensitivityTable,
    budget_bytes: float,
    non_targeted_bytes: float,
    tolerance: float = 0.02,
    max_moves: int = 100_000,
) -> tuple[dict[str, str], list[dict]]:
    """Budget-aware one-rung polish of a frozen argmax layout.

    Enumerates single rung moves (up or down) for every tensor, ranked by
    benefit-per-byte from the sensitivity table, applied greedily while the
    total stays inside the budget's dead zone. Down-moves are allowed even at
    zero measured benefit when they free bytes (they fund later up-moves),
    with a small preference for high-byte low-benefit tensors — precisely the
    "Q5_K attention rung worth nothing" swap the V50 analysis called for.

    Returns (new_layout, applied_moves).
    """
    layout = dict(layout)
    moves: list[dict] = []
    rank = {t: [q for q in QUANT_LADDER if q in table.candidates.get(t, [])] for t in layout}

    def total() -> float:
        return non_targeted_bytes + sum(table.bytes_[t][q] for t, q in layout.items())

    target = budget_bytes
    floor = target * (1.0 - tolerance)
    ceiling = target * (1.0 + tolerance)

    # Iterative passes: rank once per pass, apply while budget allows.
    for _pass in range(len(layout) + 1):
        ups, downs = [], []
        for t, q in layout.items():
            lad = rank[t]
            if not lad or q not in lad:
                continue
            i = lad.index(q)
            for j, direction in ((i + 1, "up"), (i - 1, "down")):
                if j < 0 or j >= len(lad):
                    continue
                nxt = lad[j]
                s_now, s_next = table.sens[t].get(q), table.sens[t].get(nxt)
                if s_now is None or s_next is None:
                    continue
                cost = table.bytes_[t][nxt] - table.bytes_[t][q]
                if direction == "up":
                    benefit = s_now - s_next
                    if cost > 0 and benefit > 0:
                        ups.append((benefit / cost, t, nxt, cost, benefit))
                else:
                    loss = s_next - s_now
                    freed = -cost
                    if freed > 0:
                        downs.append((loss / freed, t, nxt, cost, loss))
        if not ups and not downs:
            break
        ups.sort(key=lambda m: -m[0])      # best quality-per-byte first
        downs.sort(key=lambda m: m[0])     # least damage per freed byte first
        applied_any = False

        # 1) if over budget, take least-damaging down-moves until inside
        while total() > ceiling and downs:
            _, t, nxt, cost, _loss = downs.pop(0)
            if layout.get(t) != nxt:
                layout[t] = nxt
                moves.append({"tensor": t, "to": nxt, "dir": "down", "bytes": cost})
                applied_any = True

        # 2) swap worthless rungs down to fund better buys: pair each down
        #    with the best up that only fits after the freed bytes land
        for d_score, d_t, d_nxt, d_cost, _l in list(downs):
            if total() > ceiling:
                break
            for u in ups:
                u_score, u_t, u_nxt, u_cost, _b = u
                if u_t == d_t or layout.get(u_t) == u_nxt:
                    continue
                if total() + d_cost + u_cost <= ceiling:
                    layout[d_t] = d_nxt
                    moves.append({"tensor": d_t, "to": d_nxt, "dir": "down", "bytes": d_cost})
                    layout[u_t] = u_nxt
                    moves.append({"tensor": u_t, "to": u_nxt, "dir": "up", "bytes": u_cost})
                    applied_any = True
                    break
            if applied_any:
                break

        # 3) spend remaining slack on plain up-moves
        for u_score, t, nxt, cost, _b in ups:
            if total() + cost <= ceiling:
                if layout.get(t) != nxt:
                    layout[t] = nxt
                    moves.append({"tensor": t, "to": nxt, "dir": "up", "bytes": cost})
                    applied_any = True

        if not applied_any or len(moves) >= max_moves:
            break

    return layout, moves

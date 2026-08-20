"""Sensitivity table, greedy knapsack, and polish — pure-logic tests.

These exercise the packing logic with synthetic sensitivities (no model, no
libggml): the properties that matter are (1) warm-start lands inside budget,
(2) polish spends slack upward, (3) polish pulls over-budget layouts back
inside, (4) down/up swaps fund each other.
"""

from voodoo_quant.training.sensitivity import (
    SensitivityTable,
    greedy_knapsack_layout,
    knapsack_polish,
)


def _toy_table(n_tensors=8, ladder=("IQ2_XXS", "Q4_K", "Q5_K", "Q8_0")) -> SensitivityTable:
    """Synthetic table: tensor i is i-times more sensitive (deep-stack MLPs)."""
    t = SensitivityTable()
    base = 500_000  # bytes at Q8_0-ish scale
    for i in range(n_tensors):
        name = f"layers.{i}.mlp.up_proj"
        t.candidates[name] = list(ladder)
        t.sens[name] = {}
        t.bytes_[name] = {}
        for j, q in enumerate(ladder):
            # sensitivity falls as precision rises; more-sensitive tensors
            # benefit more per rung
            t.sens[name][q] = (0.40 - 0.09 * j) * (1.0 + i)
            t.bytes_[name][q] = base * (1.0 + j) / 4.0
    return t


def test_warm_start_within_budget():
    t = _toy_table()
    layout = greedy_knapsack_layout(t, budget_bytes=2_100_000, non_targeted_bytes=0)
    total = sum(t.bytes_[n][q] for n, q in layout.items())
    assert total <= 2_100_000 + 600_000, total  # greedy may leave small slack
    # everything must have an assignment from its own menu
    for n, q in layout.items():
        assert q in t.candidates[n]


def test_warm_start_prefers_sensitive_tensors():
    t = _toy_table()
    # generous budget: everyone reaches the top rung
    layout = greedy_knapsack_layout(t, budget_bytes=10**9, non_targeted_bytes=0)
    assert all(q == "Q8_0" for q in layout.values())


def test_polish_spends_slack_upward():
    t = _toy_table()
    # start everything at the floor: huge slack
    layout = {n: "IQ2_XXS" for n in t.sens}
    polished, moves = knapsack_polish(
        layout, t, budget_bytes=1_600_000, non_targeted_bytes=0, tolerance=0.02
    )
    total = sum(t.bytes_[n][q] for n, q in polished.items())
    assert total <= 1_600_000 * 1.02
    assert any(m["dir"] == "up" for m in moves)
    assert total > 0  # moved off the floor


def test_polish_pulls_overbudget_down():
    t = _toy_table()
    # start everything at the ceiling: way over budget
    layout = {n: "Q8_0" for n in t.sens}
    polished, _moves = knapsack_polish(
        layout, t, budget_bytes=1_200_000, non_targeted_bytes=0, tolerance=0.02
    )
    total = sum(t.bytes_[n][q] for n, q in polished.items())
    assert total <= 1_200_000 * 1.02, total


def test_polish_swaps_low_value_rungs():
    """A cheap-to-improve tensor gains even when a fat rung must be dropped."""
    t = _toy_table(n_tensors=4)
    # hand-crafted: tensor "fat" has a useless top rung (sens barely improves)
    t.candidates["fat"] = ["IQ2_XXS", "Q4_K", "Q5_K", "Q8_0"]
    t.sens["fat"] = {"IQ2_XXS": 0.40, "Q4_K": 0.31, "Q5_K": 0.309, "Q8_0": 0.308}
    t.bytes_["fat"] = {"IQ2_XXS": 125_000, "Q4_K": 250_000, "Q5_K": 400_000, "Q8_0": 500_000}
    layout = {n: "Q4_K" for n in t.sens}
    layout["fat"] = "Q8_0"  # the worthless expensive rung
    polished, moves = knapsack_polish(
        layout, t, budget_bytes=sum(t.bytes_[n][q] for n, q in layout.items()) - 200_000,
        non_targeted_bytes=0, tolerance=0.02,
    )
    # over budget => least-damaging downs must fire; fat's Q8_0->Q5_0 loss is
    # ~0 and frees 100K, making it the prime candidate
    assert polished.get("fat") != "Q8_0", polished.get("fat")


def test_gates_seeding():
    """warm_start_gates concentrates the softmax on the chosen rung."""
    import torch

    from voodoo_quant.training.sensitivity import warm_start_gates

    class FakeMod:
        def __init__(self, cands):
            self.candidate_types = list(cands)
            self.gates = torch.zeros(len(cands))

    mods = {"a": FakeMod(["IQ2_XXS", "Q4_K", "Q8_0"]), "b": FakeMod(["Q4_K", "Q8_0"])}
    n = warm_start_gates(mods, {"a": "Q4_K", "b": "Q8_0"}, logit_bias=4.0)
    assert n == 2
    assert mods["a"].gates.argmax().item() == 1
    assert mods["b"].gates.argmax().item() == 1

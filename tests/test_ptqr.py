"""Gate tests for Per-Token Quant Routing (PTQR).

Proves, on small random layers, that the PTQR forward computes each token's
output from exactly ONE candidate (no averaging anywhere), that token routing
shares track the gate probabilities, and that backward reproduces the exact
straight-through mixture gradient (grad_p_k = <G, x @ W_k^T>, grad_x through
the soft mixture weight).

Covers the routed paths added for PTQR:
  * _sample_token_routing
  * _RoutedMixedLinear            (standard linear forward)
  * _RoutedEmbeddingRowFunction   (non-lazy embedding lookup)
  * _MixedEmbeddingRowFunctionLazy with routing (lazy row-select, CUDA only)
  * MixedQuantLinear / MixedQuantEmbedding module-level PTQR toggle

Skips when libggml-base is unavailable.
"""

import pytest
import torch

from voodoo_quant import ggml

requires_ggml = pytest.mark.skipif(ggml.find_libggml() is None, reason="libggml-base not built")


@requires_ggml
class TestPTQR:
    def test_router_shares(self):
        from voodoo_quant.layers import _sample_token_routing

        torch.manual_seed(0)
        for probs_vec in ([0.25, 0.75], [0.1, 0.2, 0.3, 0.4], [0.05, 0.05, 0.9]):
            probs = torch.tensor(probs_vec, dtype=torch.float32)
            routing = _sample_token_routing(probs, 200_000)
            for k, p in enumerate(probs_vec):
                share = (routing == k).float().mean().item()
                assert abs(share - p) <= 0.01, f"p={probs_vec} share[{k}]={share:.4f}"
        # Deterministic given the seed (same draw twice under manual_seed).
        torch.manual_seed(0)
        a = _sample_token_routing(torch.tensor([0.5, 0.5]), 64)
        torch.manual_seed(0)
        b = _sample_token_routing(torch.tensor([0.5, 0.5]), 64)
        assert torch.equal(a, b)

    def test_routed_linear(self):
        from voodoo_quant.ggml import dequantize_tensor, quantize_tensor
        from voodoo_quant.layers import _RoutedMixedLinear, _sample_token_routing

        torch.manual_seed(7)
        out_f, in_f, padded = 8, 256, 256
        qts = ["IQ2_XXS", "IQ3_XXS", "Q4_K", "Q8_0"]
        weight = torch.randn(out_f, padded, dtype=torch.float32)
        qbytes = [quantize_tensor(weight, qt, None) for qt in qts]
        cands = [
            dequantize_tensor(qb, qt, out_f, padded)[:, :in_f].contiguous()
            for qb, qt in zip(qbytes, qts)
        ]
        base = torch.softmax(torch.randn(len(qts)), dim=0)
        probs = base.detach().clone().requires_grad_(True)
        x = torch.randn(2, 5, in_f, dtype=torch.float32)  # 3-D input exercises reshape
        routing = _sample_token_routing(probs.detach(), 10)

        y = _RoutedMixedLinear.apply(torch.float32, probs, routing, x, *cands)

        # Forward: every token's row equals ITS routed candidate's output exactly
        # (within matmul accumulation-order ulps: grouped [n_k,in]@ vs single
        # [1,in]@ tile differently; cross-candidate differences are ~1e-2).
        x2 = x.reshape(-1, in_f)
        y2 = y.reshape(-1, y.shape[-1])  # y keeps the 3-D input shape [2, 5, out]
        for t in range(10):
            ref = x2[t] @ cands[int(routing[t])].t()
            assert torch.allclose(y2[t], ref, rtol=1e-5, atol=1e-4), (
                f"row {t} not from routed candidate "
                f"(max|diff|={(y2[t] - ref).abs().max().item():.3e})"
            )
            others = [c for k, c in enumerate(cands) if k != int(routing[t])]
            assert not any(
                torch.allclose(y2[t], x2[t] @ c.t(), rtol=1e-5, atol=1e-4) for c in others
            ), f"row {t} matches a NON-routed candidate"

        # Backward: ST mixture gradient.
        G = torch.randn_like(y)
        y.backward(G)
        G2 = G.reshape(-1, G.shape[-1])
        gp_ref = torch.stack(
            [(G2 * (x2 @ c.t())).sum() for c in cands]
        )  # <G, x @ W_k^T> elementwise
        assert torch.allclose(probs.grad, gp_ref, rtol=1e-5, atol=1e-5), (
            f"grad_p max|diff|={(probs.grad - gp_ref).abs().max().item():.3e}"
        )

    def test_routed_embedding_rows(self):
        from voodoo_quant.ggml import dequantize_tensor, quantize_tensor
        from voodoo_quant.layers import _RoutedEmbeddingRowFunction, _sample_token_routing

        torch.manual_seed(11)
        vocab, dim = 64, 256
        qts = ["IQ2_XXS", "Q4_K"]
        weight = torch.randn(vocab, dim, dtype=torch.float32)
        qbytes = [quantize_tensor(weight, qt, None) for qt in qts]
        cands = [
            dequantize_tensor(qb, qt, vocab, dim).contiguous() for qb, qt in zip(qbytes, qts)
        ]
        base = torch.softmax(torch.randn(len(qts)), dim=0)
        probs = base.detach().clone().requires_grad_(True)
        ids = torch.randint(0, vocab, (12,))
        routing = _sample_token_routing(probs.detach(), 12)

        y = _RoutedEmbeddingRowFunction.apply(torch.float32, probs, routing, ids, *cands)

        ref = torch.stack([cands[int(routing[t])][ids[t]] for t in range(12)])
        assert torch.allclose(y, ref, rtol=0, atol=0)
        G = torch.randn_like(y)
        y.backward(G)
        gp_ref = torch.stack([(G * c[ids]).sum() for c in cands])
        assert torch.allclose(probs.grad, gp_ref, rtol=1e-5, atol=1e-5), (
            f"emb grad_p max|diff|={(probs.grad - gp_ref).abs().max().item():.3e}"
        )

    def test_module_toggle(self):
        from voodoo_quant.layers import (
            MixedQuantEmbedding,
            MixedQuantLinear,
            set_ptqr,
        )

        torch.manual_seed(23)
        in_f, out_f = 256, 8
        qts = ["IQ2_XXS", "IQ3_XXS", "Q8_0"]
        lin = MixedQuantLinear(
            in_features=in_f,
            out_features=out_f,
            candidate_types=qts,
            source_weight=torch.randn(out_f, in_f, dtype=torch.float32),
            tensor_name="tests.ptqr.linear",
        )
        emb = MixedQuantEmbedding(
            num_embeddings=64,
            embedding_dim=in_f,
            candidate_types=qts,
            source_weight=torch.randn(64, in_f, dtype=torch.float32),
            tensor_name="tests.ptqr.embedding",
        )
        x = torch.randn(4, in_f)
        ids = torch.randint(0, 64, (6,))
        cands_lin = [getattr(lin, f"_w_{k}")[:, :in_f] for k in range(len(qts))]
        cands_emb = [getattr(emb, f"_w_{k}")[:, :in_f] for k in range(len(qts))]

        set_ptqr(True)
        try:
            y = lin(x)
            probs = lin.get_probs().detach()
            outs = [x @ c.t() for c in cands_lin]
            for t in range(x.shape[0]):
                # ulp tolerance for matmul accumulation order; non-routed candidates
                # must differ by orders of magnitude more (~1e-2).
                matches = [torch.allclose(y[t], o[t], rtol=1e-5, atol=1e-4) for o in outs]
                assert sum(matches) == 1, "some row not from exactly one candidate"

            # gates.grad equals the ST mixture gradient through softmax.
            lin.zero_grad()
            y2 = lin(x)
            G = torch.randn_like(y2)
            y2.backward(G)
            gp = torch.stack([(G * (x @ c.t())).sum() for c in cands_lin])
            p = probs.clone()
            jac = torch.diag(p) - torch.outer(p, p)  # d softmax / d gates
            gates_ref = jac @ gp
            assert torch.allclose(lin.gates.grad, gates_ref, rtol=1e-4, atol=1e-4), (
                f"gates.grad max|diff|={(lin.gates.grad - gates_ref).abs().max().item():.3e}"
            )

            ey = emb(ids)
            outs_e = [c[ids] for c in cands_emb]
            for t in range(ids.numel()):
                matches = [torch.allclose(ey[t], o[t], rtol=1e-5, atol=1e-4) for o in outs_e]
                assert sum(matches) == 1, "some embedding row not from exactly one candidate"
        finally:
            set_ptqr(False)

        # Toggle off restores the mixture forward exactly.
        y_mix = lin(x)
        p = lin.get_probs().detach()
        mix = sum(p[k] * cands_lin[k] for k in range(len(qts)))
        assert torch.allclose(y_mix, x @ mix.t(), rtol=1e-5, atol=1e-5)

    def test_lazy_embedding_routing_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        from voodoo_quant.ggml import dequantize_tensor
        from voodoo_quant.layers import MixedQuantEmbedding, set_ptqr

        torch.manual_seed(31)
        dev = torch.device("cuda")
        vocab, dim = 64, 256
        qts = ["Q4_K", "Q8_0"]
        emb = MixedQuantEmbedding(
            num_embeddings=vocab,
            embedding_dim=dim,
            candidate_types=qts,
            source_weight=torch.randn(vocab, dim, dtype=torch.float32),
            device=dev,
            lazy=True,
            qbytes_device=dev,
            tensor_name="tests.ptqr.embedding_lazy",
        )
        cands = [
            dequantize_tensor(getattr(emb, f"_qweight_{k}"), qt, vocab, dim).to(dev)
            for k, qt in enumerate(qts)
        ]
        ids = torch.randint(0, vocab, (9,), device=dev)
        set_ptqr(True)
        try:
            y = emb(ids)  # [9, dim]
            for t in range(ids.numel()):
                matches = [torch.allclose(y[t], c[ids[t]], rtol=1e-2, atol=1e-2) for c in cands]
                assert sum(matches) >= 1, "lazy routed embedding row not from any single candidate"
        finally:
            set_ptqr(False)

"""
GPU-side dequantization for IQ quant types using PyTorch-native operations.

All IQ types use lookup tables (LUTs) that are uploaded to GPU as tensors.
The quantized bytes index into these LUTs to reconstruct fp32 values.

LUTs are loaded once and cached on the target device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from voodoo_quant.quants.luts import (
    IQ2XXS_GRID, IQ2XS_GRID, IQ2S_GRID,
    IQ3XXS_GRID, IQ3S_GRID, IQ1S_GRID,
    KSIGNS_IQ2XS, KMASK_IQ2XS, KVALUES_IQ4NL,
)

QK_K = 256
IQ1S_DELTA = 0.125

# Cache for GPU tensors
_LUT_CACHE: dict[str, torch.Tensor] = {}


def _get_lut(name: str, data: list, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Get or create a cached GPU tensor for a lookup table."""
    key = f"{name}_{device}_{dtype}"
    if key not in _LUT_CACHE:
        # For uint64 data that overflows signed int64, use numpy as intermediary
        import numpy as np
        arr = np.array(data, dtype=np.uint64 if dtype == torch.int64 else None)
        if dtype == torch.int64:
            # Interpret as signed int64 (bit pattern preserved)
            t = torch.from_numpy(arr.astype(np.int64)).to(device)
        else:
            t = torch.tensor(data, dtype=dtype, device=device)
        _LUT_CACHE[key] = t
    return _LUT_CACHE[key]


def _f16_bytes_to_f32(raw_bytes: torch.Tensor) -> torch.Tensor:
    """Convert [N, 2] uint8 bytes to [N] float32 (little-endian fp16)."""
    return raw_bytes.contiguous().view(torch.float16).squeeze(-1).to(torch.float32)


# ---------------------------------------------------------------------------
# IQ2_XXS: block = d(2) + qs[QK_K/8 * uint16] = 2 + 64 = 66 bytes
# ---------------------------------------------------------------------------

def _dequantize_iq2_xxs_gpu(qbytes_gpu, out_features, in_features):
    block_bytes = 66
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = _f16_bytes_to_f32(raw[:, :2])

    # qs: QK_K/8 = 32 uint16 values = 64 bytes
    # Reinterpret as uint16
    qs_bytes = raw[:, 2:66].contiguous()  # [nblocks, 64]
    qs_u16 = qs_bytes.view(torch.int16).view(torch.uint16)  # [nblocks, 32]

    # iq2xxs_grid: 256 entries of uint64, each contains 8 bytes
    # ksigns_iq2xs: 128 entries of uint8
    # kmask_iq2xs: [1,2,4,8,16,32,64,128]
    grid = _get_lut("iq2xxs_grid", IQ2XXS_GRID, torch.int64, qbytes_gpu.device)
    ksigns = _get_lut("ksigns_iq2xs", KSIGNS_IQ2XS, torch.uint8, qbytes_gpu.device)
    kmask = _get_lut("kmask_iq2xs", KMASK_IQ2XS, torch.uint8, qbytes_gpu.device)

    # Reinterpret int64 grid as uint8 bytes: each int64 = 8 bytes
    grid_u8 = grid.view(torch.uint8).view(-1).reshape(-1, 8)  # [N, 8]

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib32 in range(QK_K // 32):  # 8 iterations
        # Each ib32 processes 4 * 8 = 32 weights
        # aux32[0] = qs[4*ib32], aux32[1] = qs[4*ib32+1] (as uint32 pairs)
        # Actually: memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t))
        # So aux32[0] = first 4 bytes, aux32[1] = next 4 bytes
        # qs_u16 has 32 uint16 per block. 4*ib32 indexes into uint16, so byte offset = 8*ib32
        # aux32[0] = bytes[8*ib32 : 8*ib32+4] as uint32
        # aux32[1] = bytes[8*ib32+4 : 8*ib32+8] as uint32

        byte_offset = 8 * ib32
        aux32_0 = qs_bytes[:, byte_offset:byte_offset+4].contiguous().view(torch.int32).to(torch.int32).squeeze(-1)
        aux32_1 = qs_bytes[:, byte_offset+4:byte_offset+8].contiguous().view(torch.int32).to(torch.int32).squeeze(-1)

        # Convert to int64 for unsigned shift operations (int32 arithmetic shift is wrong for negative values)
        aux32_1_u = aux32_1.to(torch.int64) & 0xFFFFFFFF
        db = d * (0.5 + (aux32_1_u >> 28).to(torch.float32)) * 0.25

        # aux8 is the byte view of aux32 (both 0 and 1 concatenated = 8 bytes)
        # aux8[l] for l=0..3 is the first 4 bytes (from aux32[0])
        aux8 = qs_bytes[:, byte_offset:byte_offset+8].to(torch.int32)  # [nblocks, 8]

        for l in range(4):
            grid_idx = torch.select(aux8, 1, l).long()  # [nblocks] as int64
            grid_vals = grid_u8[grid_idx]  # [nblocks, 8]

            # signs index: (aux32[1] >> (7*l)) & 127
            signs_idx = ((aux32_1_u >> (7 * l)) & 127).long()
            signs_val = ksigns[signs_idx]  # [nblocks]

            # Apply sign mask using arithmetic: 1 - 2*(bit_set)
            # This avoids torch.where shape issues entirely
            kmask_t = kmask.to(torch.int32)
            sign_expanded = signs_val.unsqueeze(1)  # [nblocks, 1]
            sign_bits = 1.0 - 2.0 * ((sign_expanded & kmask_t.unsqueeze(0)) != 0).to(torch.float32)  # [nblocks, 8]
            vals = db.unsqueeze(1) * grid_vals.to(torch.float32) * sign_bits  # [nblocks, 8]
            y_start = ib32 * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ2_XS: block = d(2) + qs[QK_K/8 * uint16] + scales[QK_K/32] = 2 + 64 + 8 = 74 bytes
# ---------------------------------------------------------------------------

def _dequantize_iq2_xs_gpu(qbytes_gpu, out_features, in_features):
    block_bytes = 74
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = _f16_bytes_to_f32(raw[:, :2])
    qs_bytes = raw[:, 2:66]  # 64 bytes = 32 uint16
    scales = raw[:, 66:74]   # 8 bytes

    grid = _get_lut("iq2xs_grid", IQ2XS_GRID, torch.int64, qbytes_gpu.device)
    ksigns = _get_lut("ksigns_iq2xs", KSIGNS_IQ2XS, torch.uint8, qbytes_gpu.device)
    kmask = _get_lut("kmask_iq2xs", KMASK_IQ2XS, torch.uint8, qbytes_gpu.device)
    grid_u8 = grid.view(torch.uint8).view(-1).reshape(-1, 8)  # [N, 8]

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib32 in range(QK_K // 32):  # 8 iterations
        sc = scales[:, ib32].to(torch.int32)
        db0 = d * (0.5 + (sc & 0xf).to(torch.float32)) * 0.25
        db1 = d * (0.5 + (sc >> 4).to(torch.float32)) * 0.25

        for l in range(4):
            # qs[4*ib32 + l] is a uint16
            qs_idx = qs_bytes[:, (4*ib32 + l)*2:(4*ib32 + l)*2+2].contiguous().view(torch.int16).view(torch.uint16).squeeze(-1).to(torch.int64)

            grid_idx = qs_idx & 511
            signs_idx = (qs_idx >> 9).to(torch.int64)

            grid_vals = grid_u8[grid_idx]  # [nblocks, 8]
            signs_val = ksigns[signs_idx]  # [nblocks]

            # Apply sign mask using arithmetic
            kmask_t = kmask.to(torch.int32)
            sign_expanded = signs_val.unsqueeze(1)
            sign_bits = 1.0 - 2.0 * ((sign_expanded & kmask_t.unsqueeze(0)) != 0).to(torch.float32)

            db = db0 if l < 2 else db1
            vals = db.unsqueeze(1) * grid_vals.to(torch.float32) * sign_bits
            y_start = ib32 * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ2_S: block = d(2) + qs[QK_K/4] + qh[QK_K/32] + scales[QK_K/32] = 2 + 64 + 8 + 8 = 82 bytes
# ---------------------------------------------------------------------------

def _dequantize_iq2_s_gpu(qbytes_gpu, out_features, in_features):
    block_bytes = 82
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = _f16_bytes_to_f32(raw[:, :2])
    qs = raw[:, 2:66]       # 64 bytes
    qh = raw[:, 66:74]      # 8 bytes
    scales = raw[:, 74:82]  # 8 bytes

    grid = _get_lut("iq2s_grid", IQ2S_GRID, torch.int64, qbytes_gpu.device)
    ksigns = _get_lut("ksigns_iq2xs", KSIGNS_IQ2XS, torch.uint8, qbytes_gpu.device)
    kmask = _get_lut("kmask_iq2xs", KMASK_IQ2XS, torch.uint8, qbytes_gpu.device)
    grid_u8 = grid.view(torch.uint8).view(-1).reshape(-1, 8)  # [N, 8]

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib32 in range(QK_K // 32):  # 8 iterations
        sc = scales[:, ib32].to(torch.int32)
        db0 = d * (0.5 + (sc & 0xf).to(torch.float32)) * 0.25
        db1 = d * (0.5 + (sc >> 4).to(torch.float32)) * 0.25

        qh_byte = qh[:, ib32].to(torch.int32)
        signs_base = ib32 * 4  # signs come from qs + QK_K/8

        for l in range(4):
            qs_val = qs[:, ib32*4 + l].to(torch.int32)
            grid_idx = (qs_val | ((qh_byte << (8 - 2*l)) & 0x300)).to(torch.int64)

            grid_vals = grid_u8[grid_idx]
            # signs are at qs[QK_K/8 + ib32*4 + l] = qs[32 + ib32*4 + l]
            # IQ2_S uses raw sign bytes directly (NOT through ksigns_iq2xs lookup)
            signs_val = qs[:, 32 + ib32*4 + l].to(torch.int32)

            # Apply sign mask using arithmetic
            kmask_t = kmask.to(torch.int32)
            sign_expanded = signs_val.unsqueeze(1)
            sign_bits = 1.0 - 2.0 * ((sign_expanded & kmask_t.unsqueeze(0)) != 0).to(torch.float32)

            db = db0 if l < 2 else db1
            vals = db.unsqueeze(1) * grid_vals.to(torch.float32) * sign_bits
            y_start = ib32 * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ3_XXS: block = d(2) + qs[3*QK_K/8] = 2 + 96 = 98 bytes
# ---------------------------------------------------------------------------

def _dequantize_iq3_xxs_gpu(qbytes_gpu, out_features, in_features):
    block_bytes = 98
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = _f16_bytes_to_f32(raw[:, :2])
    qs = raw[:, 2:98]  # 96 bytes

    grid = _get_lut("iq3xxs_grid", IQ3XXS_GRID, torch.int32, qbytes_gpu.device)
    ksigns = _get_lut("ksigns_iq2xs", KSIGNS_IQ2XS, torch.uint8, qbytes_gpu.device)
    kmask = _get_lut("kmask_iq2xs", KMASK_IQ2XS, torch.uint8, qbytes_gpu.device)
    # iq3xxs_grid is uint32, view as 4 uint8 values
    grid_u8 = grid.view(torch.uint8).view(-1).reshape(-1, 4)  # [N, 4]

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib32 in range(QK_K // 32):  # 8 iterations
        # scales_and_signs = qs + QK_K/4 = qs + 64
        sa_offset = 64 + 4 * ib32
        sa_bytes = qs[:, sa_offset:sa_offset+4].contiguous()
        aux32 = sa_bytes.view(torch.int32).to(torch.int32).squeeze(-1)
        aux32_u = aux32.to(torch.int64) & 0xFFFFFFFF

        db = d * (0.5 + (aux32_u >> 28).to(torch.float32)) * 0.5

        qs_base = ib32 * 8
        for l in range(4):
            signs_idx = ((aux32_u >> (7 * l)) & 127).long()
            signs_val = ksigns[signs_idx]

            grid1_idx = torch.select(qs, 1, qs_base + 2*l).long()
            grid2_idx = torch.select(qs, 1, qs_base + 2*l + 1).long()
            grid1_vals = grid_u8[grid1_idx]  # [nblocks, 4]
            grid2_vals = grid_u8[grid2_idx]

            # Apply sign mask using arithmetic (4+4 = 8 values)
            kmask_t = kmask.to(torch.int32)
            sign_expanded = signs_val.unsqueeze(1)
            sign_bits = 1.0 - 2.0 * ((sign_expanded & kmask_t.unsqueeze(0)) != 0).to(torch.float32)

            vals = torch.cat([grid1_vals, grid2_vals], dim=1).to(torch.float32) * db.unsqueeze(1) * sign_bits
            y_start = ib32 * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ3_S: block = d(2) + qs[QK_K/4] + qh[QK_K/32] + signs[QK_K/8] + scales[QK_K/64]
#       = 2 + 64 + 8 + 32 + 4 = 110 bytes
# ---------------------------------------------------------------------------

def _dequantize_iq3_s_gpu(qbytes_gpu, out_features, in_features):
    IQ3S_N_SCALE = QK_K // 64  # 4
    block_bytes = 2 + QK_K//4 + QK_K//32 + QK_K//8 + IQ3S_N_SCALE  # 2+64+8+32+4 = 110
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = _f16_bytes_to_f32(raw[:, :2])
    qs = raw[:, 2:66]       # 64 bytes
    qh = raw[:, 66:74]      # 8 bytes
    signs_data = raw[:, 74:106]  # 32 bytes
    scales = raw[:, 106:110]     # 4 bytes

    grid = _get_lut("iq3s_grid", IQ3S_GRID, torch.int32, qbytes_gpu.device)
    ksigns = _get_lut("ksigns_iq2xs", KSIGNS_IQ2XS, torch.uint8, qbytes_gpu.device)
    kmask = _get_lut("kmask_iq2xs", KMASK_IQ2XS, torch.uint8, qbytes_gpu.device)
    grid_u8 = grid.view(torch.uint8).view(-1).reshape(-1, 4)  # [N, 4]

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    # Process 2 ib32 at a time
    for ib32 in range(0, QK_K // 32, 2):
        sc = scales[:, ib32 // 2].to(torch.int32)
        db1 = d * (1 + 2 * (sc & 0xf).to(torch.float32))
        db2 = d * (1 + 2 * (sc >> 4).to(torch.float32))

        qh_idx = ib32 // 2 * 2  # qh advances by 2 per pair

        for l in range(4):
            qs_base = (ib32 - qh_idx) * 4 + l  # Wait, this needs fixing
            pass

        # Simpler approach: follow the C code exactly
        # qs pointer is at qs + ib32*4 initially for first ib32, then qs += 8
        # qh pointer is at qh + ib32 (advances by 2 per pair)
        # signs pointer at signs + ib32*4 (advances by 8 per pair)

        qs_offset_1 = ib32 * 8  # each ib32 consumes 8 bytes of qs (2 bytes per l, 4 l values)
        signs_offset_1 = ib32 * 4  # each ib32 consumes 4 bytes of signs
        qh_byte_1 = qh[:, ib32 // 2 * 2].to(torch.int32)

        for l in range(4):
            grid1_idx = (qs[:, qs_offset_1 + 2*l].to(torch.int32) | ((qh_byte_1 << (8-2*l)) & 256)).to(torch.int64)
            grid2_idx = (qs[:, qs_offset_1 + 2*l + 1].to(torch.int32) | ((qh_byte_1 << (7-2*l)) & 256)).to(torch.int64)
            grid1_vals = grid_u8[grid1_idx]
            grid2_vals = grid_u8[grid2_idx]

            signs_val = signs_data[:, signs_offset_1 + l].to(torch.int32)
            kmask_t = kmask.to(torch.int32)
            sign_expanded = signs_val.unsqueeze(1)
            sign_bits = 1.0 - 2.0 * ((sign_expanded & kmask_t.unsqueeze(0)) != 0).to(torch.float32)

            vals = torch.cat([grid1_vals, grid2_vals], dim=1).to(torch.float32) * db1.unsqueeze(1) * sign_bits
            y_start = ib32 * 32 + l * 8
            y[:, y_start:y_start+8] = vals

        # Second ib32 in pair
        qs_offset_2 = qs_offset_1 + 8
        signs_offset_2 = signs_offset_1 + 4
        qh_byte_2 = qh[:, ib32 // 2 * 2 + 1].to(torch.int32)

        for l in range(4):
            grid1_idx = (qs[:, qs_offset_2 + 2*l].to(torch.int32) | ((qh_byte_2 << (8-2*l)) & 256)).to(torch.int64)
            grid2_idx = (qs[:, qs_offset_2 + 2*l + 1].to(torch.int32) | ((qh_byte_2 << (7-2*l)) & 256)).to(torch.int64)
            grid1_vals = grid_u8[grid1_idx]
            grid2_vals = grid_u8[grid2_idx]

            signs_val = signs_data[:, signs_offset_2 + l].to(torch.int32)
            kmask_t = kmask.to(torch.int32)
            sign_expanded = signs_val.unsqueeze(1)
            sign_bits = 1.0 - 2.0 * ((sign_expanded & kmask_t.unsqueeze(0)) != 0).to(torch.float32)

            vals = torch.cat([grid1_vals, grid2_vals], dim=1).to(torch.float32) * db2.unsqueeze(1) * sign_bits
            y_start = (ib32+1) * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ1_S: block = d(2) + qs[QK_K/8] + qh[QK_K/32 * uint16] = 2 + 32 + 16 = 50 bytes
# ---------------------------------------------------------------------------

def _dequantize_iq1_s_gpu(qbytes_gpu, out_features, in_features):
    block_bytes = 50
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    d = _f16_bytes_to_f32(raw[:, :2])
    qs = raw[:, 2:34]    # 32 bytes
    qh_bytes = raw[:, 34:50]  # 16 bytes = 8 uint16

    # iq1s_grid: 2048 entries of uint64, view as 8 int8 values
    grid = _get_lut("iq1s_grid", IQ1S_GRID, torch.int64, qbytes_gpu.device)
    grid_i8 = grid.view(torch.int8).view(-1).reshape(-1, 8)  # [N, 8]

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib in range(QK_K // 32):  # 8 iterations
        qh_val = qh_bytes[:, ib*2:ib*2+2].contiguous().view(torch.int16).view(torch.uint16).squeeze(-1).to(torch.int32)
        dl = d * (2 * ((qh_val >> 12) & 7) + 1).to(torch.float32)
        delta = torch.where((qh_val & 0x8000) != 0, -IQ1S_DELTA, IQ1S_DELTA)

        for l in range(4):
            qs_val = qs[:, ib*4 + l].to(torch.int32)
            grid_idx = (qs_val | (((qh_val >> (3*l)) & 7) << 8)).to(torch.int64)
            grid_vals = grid_i8[grid_idx]  # [nblocks, 8]

            vals = dl.unsqueeze(1) * (grid_vals.to(torch.float32) + delta.unsqueeze(1))
            y_start = ib * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ1_M: block = qs[QK_K/8] + qh[QK_K/16] + scales[QK_K/32]
#       = 32 + 16 + 8 = 56 bytes (NO separate d field — d is packed in scales)
# ---------------------------------------------------------------------------

def _dequantize_iq1_m_gpu(qbytes_gpu, out_features, in_features):
    block_bytes = 56
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)
    # Struct: qs[32] + qh[16] + scales[8]
    qs = raw[:, 0:32]
    qh = raw[:, 32:48]
    scales = raw[:, 48:56]

    grid = _get_lut("iq1s_grid", IQ1S_GRID, torch.int64, qbytes_gpu.device)
    grid_i8 = grid.view(torch.int8).view(-1).reshape(-1, 8)  # [N, 8]

    # Extract d from scales (packed as uint16 in first 8 bytes)
    # scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
    # sc is uint16*, so sc[0] = first 2 bytes, sc[1] = next 2, etc.
    sc_u16 = scales.view(torch.int16).view(torch.uint16).reshape(nblocks, 4)
    sc0 = sc_u16[:, 0].to(torch.int32)
    sc1 = sc_u16[:, 1].to(torch.int32)
    sc2 = sc_u16[:, 2].to(torch.int32)
    sc3 = sc_u16[:, 3].to(torch.int32)
    scale_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000)
    # Convert to fp16 then fp32 (bit pattern interpretation)
    d = scale_u16.to(torch.int16).view(torch.float16).to(torch.float32)

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib in range(QK_K // 32):  # 8 iterations
        sc_val = sc_u16[:, ib // 2].to(torch.int32)
        dl1 = d * (2 * ((sc_val >> (6 * (ib % 2) + 0)) & 0x7) + 1).to(torch.float32)
        dl2 = d * (2 * ((sc_val >> (6 * (ib % 2) + 3)) & 0x7) + 1).to(torch.float32)

        qh_base = ib * 2
        qh0 = qh[:, qh_base].to(torch.int32)
        qh1 = qh[:, qh_base + 1].to(torch.int32)

        idx = torch.stack([
            qs[:, ib*4 + 0].to(torch.int32) | ((qh0 << 8) & 0x700),
            qs[:, ib*4 + 1].to(torch.int32) | ((qh0 << 4) & 0x700),
            qs[:, ib*4 + 2].to(torch.int32) | ((qh1 << 8) & 0x700),
            qs[:, ib*4 + 3].to(torch.int32) | ((qh1 << 4) & 0x700),
        ], dim=1)  # [nblocks, 4]

        deltas = torch.stack([
            torch.where((qh0 & 0x08) != 0, -IQ1S_DELTA, IQ1S_DELTA),
            torch.where((qh0 & 0x80) != 0, -IQ1S_DELTA, IQ1S_DELTA),
            torch.where((qh1 & 0x08) != 0, -IQ1S_DELTA, IQ1S_DELTA),
            torch.where((qh1 & 0x80) != 0, -IQ1S_DELTA, IQ1S_DELTA),
        ], dim=1)  # [nblocks, 4]

        for l in range(4):
            grid_vals = grid_i8[idx[:, l].to(torch.int64)]  # [nblocks, 8]
            dl = dl1 if l < 2 else dl2
            vals = dl.unsqueeze(1) * (grid_vals.to(torch.float32) + deltas[:, l].unsqueeze(1))
            y_start = ib * 32 + l * 8
            y[:, y_start:y_start+8] = vals

    return y.reshape(out_features, in_features)


# ---------------------------------------------------------------------------
# IQ4_XS: block = d(2) + qs[QK_K/2] + scales_l[QK_K/32] + scales_h[2] = 2 + 128 + 8 + 2 = 140 bytes
# Wait, let me check the actual struct...
# block_iq4_xs has: scales_h (uint16), scales_l[QK_K/32], qs[QK_K/2], d
# Actually from the static_assert and C code:
# dequantize_row_iq4_xs uses: x[i].scales_l, x[i].scales_h, x[i].qs, x[i].d
# ---------------------------------------------------------------------------

def _dequantize_iq4_xs_gpu(qbytes_gpu, out_features, in_features):
    # Need to find the struct layout. From ggml-common.h:
    # block_iq4_xs: I need to find it
    # Looking at the C code: x[i].scales_l[ib/2], x[i].scales_h, x[i].qs, x[i].d
    # From the quant registry: IQ4_XS has block_bytes=136, block_size=256
    block_bytes = 136
    nblocks_per_row = in_features // QK_K
    nblocks = out_features * nblocks_per_row
    raw = qbytes_gpu[:nblocks * block_bytes].reshape(nblocks, block_bytes)

    # The struct layout from ggml-common.h (searching for block_iq4_xs):
    # typedef struct {
    #     ggml_half d;
    #     uint16_t scales_h;
    #     uint8_t scales_l[QK_K/32];
    #     uint8_t qs[QK_K/2];
    # } block_iq4_xs;
    # sizeof = 2 + 2 + 8 + 128 = 140... but registry says 136
    # Let me check: maybe scales_h is only 1 byte?
    # Actually: IQ4_XS block_bytes=136 from the registry
    # 2 + 2 + 8 + 128 = 140 != 136
    # Maybe: d(2) + scales_l[8] + scales_h(?) + qs[128] = 136 -> scales_h = 136-2-8-128 = -2? No.
    # Actually looking more carefully at the C code:
    # x[i].scales_l[ib/2] with ib going 0..7, so scales_l has 4 bytes? No, 8 bytes (QK_K/32=8)
    # x[i].scales_h with 2*ib bits used (ib=0..7 -> 16 bits = 2 bytes)
    # So: d(2) + scales_h(2) + scales_l(8) + qs(128) = 140
    # But registry says 136. Let me re-check.
    # Actually, the registry in voodoo_quant.ggml says:
    # "IQ4_XS": QuantTypeInfo("IQ4_XS", 23, 256, 136, ...)
    # So block_bytes=136. Maybe scales_h is not present or is smaller.
    # Let me check the actual struct in the header

    # For now, try the layout that gives 136 bytes:
    # d(2) + scales_l[8] + qs[128] = 138, not 136
    # d(2) + qs[128] + scales_l[4] = 134, not 136
    # d(2) + scales_h(2) + scales_l[4] + qs[128] = 136! That's it.
    # But the C code uses scales_l[ib/2] for ib=0..7, so scales_l needs 4 elements (indices 0..3)
    # And scales_h provides the high bits: (scales_h >> 2*ib) & 3
    # With 4 scales_l bytes and 2 scales_h bytes: 2+2+4+128 = 136. Yes!

    d = _f16_bytes_to_f32(raw[:, :2])
    scales_h = raw[:, 2:4].contiguous().view(torch.int16).view(torch.uint16).squeeze(-1).to(torch.int32)
    scales_l = raw[:, 4:8]   # 4 bytes
    qs = raw[:, 8:136]       # 128 bytes

    kvalues = _get_lut("kvalues_iq4nl", KVALUES_IQ4NL, torch.int8, qbytes_gpu.device)

    y = torch.zeros(nblocks, QK_K, device=qbytes_gpu.device, dtype=torch.float32)

    for ib in range(QK_K // 32):  # 8 iterations
        ls = ((scales_l[:, ib // 2].to(torch.int32) >> (4 * (ib % 2))) & 0xf) | \
             (((scales_h >> (2 * ib)) & 3) << 4)
        dl = d * (ls.to(torch.float32) - 32.0)

        for j in range(16):
            q = qs[:, ib*16 + j].to(torch.int32)
            y[:, ib*32 + j] = dl * kvalues[(q & 0xf).to(torch.int64)].to(torch.float32)
            y[:, ib*32 + j + 16] = dl * kvalues[(q >> 4).to(torch.int64)].to(torch.float32)

    return y.reshape(out_features, in_features)

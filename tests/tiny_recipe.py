"""Tiny Qwen3.5-family config overrides for fast e2e/integration runs.

Used by ``VOODOO_TINY_RECIPE=1`` in the trainer: keeps the hybrid
linear/full-attention architecture and gated-q structure while shrinking
every dimension so a training run completes in seconds on small GPUs.
"""

TINY_CFG = {
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "head_dim": 64,
    "num_key_value_heads": 2,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "intermediate_size": 512,
    "vocab_size": 4096,
    "max_position_embeddings": 4096,
    "linear_conv_kernel_dim": 4,
}

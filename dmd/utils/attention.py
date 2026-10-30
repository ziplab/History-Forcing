"""Validation for causal attention windows, measured in latent frames."""


def validate_local_attn_size(value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"local_attn_size must be a positive integer, got {value!r}")
    return value

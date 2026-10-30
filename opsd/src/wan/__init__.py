"""Shared Wan video models, codecs and AR adapters with version-specific conditioning."""


def model_class(base):
    if base == 'wan2.1':
        from .wan21.model import WanModel
    elif base == 'wan2.2':
        from .wan22.model import WanModel
    else:
        raise ValueError(f'Unknown Wan base: {base!r}')
    return WanModel

"""LongLive v1.0 with its native three-frame sink."""

from .wan_ar import Action2VDMDGenerator, WanAR


class LongLive(Action2VDMDGenerator, WanAR):
    base = "wan2.1"
    num_frame_per_block = 3
    local_attn_size = 12
    sink_size = 3
    camera_control = False
    default_latent_frames = 120

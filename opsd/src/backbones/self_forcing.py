"""Self-Forcing few-step checkpoint with a bounded rolling history."""

from .wan_ar import Action2VDMDGenerator, WanAR


class SelfForcing(Action2VDMDGenerator, WanAR):
    base = "wan2.1"
    num_frame_per_block = 3
    local_attn_size = 21
    sink_size = 0
    camera_control = False
    default_latent_frames = 21

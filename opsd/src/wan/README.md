# Wan and Camera Conditioning

Wan provides the backbone, codec, text encoder and full-context or autoregressive execution. Camera conditioning supplies prepared inputs and model-specific injection. The DreamX configuration selects the released combination of these components.

## Backbones

The native presets cover Wan2.1 T2V 1.3B/14B, Wan2.1 I2V 14B, Wan2.2 TI2V 5B and Wan2.2 T2V/I2V A14B. `config.json` supplies the checkpoint's architecture; presets provide codec and sampling defaults.

- [model.py](model.py) and [layers.py](layers.py): native DiT blocks, attention and RoPE.
- [causal.py](causal.py): block-causal execution and teacher-forcing masks.
- [conditioning.py](conditioning.py): temporal inputs, prefixes and frame layout.
- [geometry.py](geometry.py): camera transforms, ray features and projective geometry.
- [kv_cache/causal.py](kv_cache/causal.py): rolling KV, sink frames and teacher cache views.
- [vae/](vae/), [t5.py](t5.py) and [tokenizers.py](tokenizers.py): image/video and text encoding.

Wan2.2 A14B retains its high/low-noise expert selection. The 16- and 48-channel codecs retain their native frame layouts and streaming decoder caches.

## Camera Inputs

Stored camera inputs use absolute W2C `viewmats`, normalized 3×3 `intrinsics` and RGB frame indices. Condition and target arrays share the video timeline. `TemporalLayout` maps this timeline to latent frames and AR blocks.

`TemporalInputs` carries named temporal and static conditions. Temporal values can be tensors or lazy ray features. Camera preparation chooses the reference pose, converts intrinsics and constructs the representation required by the mechanism. Geometry is prepared before block slicing.

## Camera Interfaces

| Operation | Existing hooks |
|---|---|
| Prepare the complete camera timeline | `prepare_inputs` |
| Construct fixed-camera teacher conditions | `fixed_view_inputs` |
| Update an interactive camera segment | `update_interactive_camera` |
| Select and install model layers | `block_type`, `attention_type`, `_init_conditioning`, `init_conditioning` |
| Align conditions with a model forward | `condition_kwargs`, `block_kwargs` |
| Inject camera information | `condition_attention`, `inject_camera`, camera attention branches |
| Declare learned modules and cache streams | `camera_modules`, `cache_streams` |

Fixed-camera conditions follow each representation's reference convention. DreamX's previous-block representation uses identity poses for the fixed teacher. Ray mechanisms retain their own coordinate and normalization rules.

## Injection Mechanisms

The implementations include parallel PRoPE attention, projective attention over visual Q/K/V, ray projection with feature modulation, pose/Plücker conditions and discrete action adapters. Geometry and input alignment are shared; each mechanism owns its learned projection, injection order and KV coordinate system.

Camera layers are registered on native models and blocks with their checkpoint parameter names. `camera_modules` controls camera freezing and LoRA selection. The model declares any additional camera KV streams; the execution layer passes temporal positions and commit state to their native implementation.

DreamX's visual attention retains its relative RoPE positions and chunk-local camera branch. Its public backend names are `backbones.dreamx:DreamXAR` and `backbones.dreamx:DreamXBidirectional`.

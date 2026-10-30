"""Camera return rotation error at the model's last control timestamp."""

from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


PROTOCOL = 'command_return_rotation_v1'
METRICS = ('ReturnRotError_deg',)


def return_rotation_error(estimated, viewmats, frame_indices):
    """Compare final commanded/estimated rotations after first-frame alignment."""
    estimated, viewmats, indices = map(np.asarray, (estimated, viewmats, frame_indices))
    if (estimated.ndim != 3 or estimated.shape[1:] != (4, 4)
            or viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4)
            or indices.shape != (len(viewmats),) or len(indices) < 2
            or not np.issubdtype(indices.dtype, np.integer)
            or indices[0] != 0 or np.any(np.diff(indices) <= 0)
            or indices[-1] >= len(estimated)
            or not np.isfinite(estimated).all() or not np.isfinite(viewmats).all()):
        raise ValueError('Invalid estimated poses or latent camera timeline')
    # MapAnything and packed camera rotations can have small numeric deviations
    # from SO(3). Project both consistently before measuring physical angles.
    reference = Rotation.from_matrix(np.linalg.inv(viewmats[[0, -1]].astype(np.float64))[:, :3, :3])
    observed = Rotation.from_matrix(estimated[indices[[0, -1]], :3, :3])
    reference, observed = (rotations[0].inv() * rotations[1] for rotations in (reference, observed))
    error = float(np.degrees((observed.inv() * reference).magnitude()))
    return {'ReturnRotError_deg': error, 'camera_anchor_count': len(indices),
            'camera_protocol': PROTOCOL}


def cached_poses(camera, frames, video, destination, device, *, estimator=None):
    """Keep full-frame upstream estimates, so metric changes need no GPU rerun."""
    stat = Path(video).stat()
    signature = {'video': str(Path(video).resolve()), 'mtime_ns': stat.st_mtime_ns,
                 'bytes': stat.st_size, 'num_frames': frames.num_frames, 'max_side': frames.max_side}
    destination = Path(destination)
    if destination.is_file():
        with np.load(destination) as cache:
            if json.loads(str(cache['signature'])) == signature:
                return cache['poses']
    estimator = estimator or camera._estimate_poses_mapanything
    poses = estimator(frames, list(range(frames.num_frames)), device)
    if (poses is None or np.shape(poses) != (frames.num_frames, 4, 4)
            or not np.isfinite(poses).all()):
        raise ValueError(f'Incomplete camera estimates: {video}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, poses=poses, signature=json.dumps(signature, sort_keys=True))
    temporary.replace(destination)
    return poses


@contextmanager
def cache_upstream_estimates(camera, video, destination):
    """Cache the upstream estimator without changing its official metric math."""
    estimator = camera._estimate_poses_mapanything

    def estimate(frames, indices, device):
        if indices != list(range(frames.num_frames)):
            raise ValueError('Expected the unchanged full-frame MemoBench estimator')
        return cached_poses(camera, frames, video, destination, device, estimator=estimator)

    with patch.object(camera, '_estimate_poses_mapanything', estimate):
        yield

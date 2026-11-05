"""Read-only float sky statistics using the actual native reader mask rule."""
import hashlib
from pathlib import Path


def sky_statistics(runtime, raw, prediction, target):
    """Add metrics only; never modify input RGB, rendering or the source mask.

    sky_RGB is per-pixel mean absolute RGB error, with pixel count/sum/mean.
    RGB channel sums/means remain unclipped float values before PNG encoding.
    """
    import cv2
    import numpy as np
    from PIL import Image

    native = getattr(runtime, 'n', None)
    if native is None:
        native = getattr(getattr(runtime, 'source', None), 'n', None)
    if native is None:
        raise ValueError('Sky statistics require runtime.n or runtime.source.n')
    source = Path(native.config['source_path'])
    path = source/'sky/sky_0'/f'0008{int(raw):03d}.png'

    def array(value):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)

    pred, truth = array(prediction), array(target)
    if pred.ndim != 3 or pred.shape[0] != 3 or pred.shape != truth.shape:
        raise ValueError('Expected matching RGB[3,H,W] for sky readout')
    height, width = pred.shape[1:]
    original = np.asarray(Image.open(path))
    if original.ndim != 2:
        raise ValueError('The native scene008 sky masks must be single-channel')
    # waymo_loader.py:465-469 thresholds BEFORE resizing. camera_utils.py:66-73
    # uses default cv2 INTER_LINEAR, and Camera.__init__ thresholds again >0.
    binary = (original > 0).astype(np.float32)
    resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_LINEAR) if binary.shape != (height, width) else binary
    mask = resized > 0
    count = int(mask.sum())
    if not count:
        raise ValueError(f'Empty actual sky mask: {path}')
    selected_pred = pred[:, mask].astype(np.float64)
    selected_truth = truth[:, mask].astype(np.float64)
    if not (np.isfinite(selected_pred).all() and np.isfinite(selected_truth).all()):
        raise ValueError('Nonfinite sky RGB')
    error = np.abs(selected_pred-selected_truth).mean(0)
    pred_sum = selected_pred.sum(1)
    truth_sum = selected_truth.sum(1)
    total = float(error.sum())
    return dict(
        sky_RGB=dict(count=count, sum=total, mean=total/count),
        sky_prediction_RGB_mean=(pred_sum/count).tolist(),
        sky_GT_RGB_mean=(truth_sum/count).tolist(),
        sky_prediction_RGB_sum=pred_sum.tolist(), sky_GT_RGB_sum=truth_sum.tolist(),
        sky_mask=dict(path=str(path), source_shape=list(original.shape), render_shape=[height, width],
                      sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                      rule='(source PNG>0).astype(float32) -> cv2.INTER_LINEAR -> >0; same native reader',
                      scope='All actual sky-mask pixels, with no actor/M2/colour selection'),
        sky_float_before_PNG=True)

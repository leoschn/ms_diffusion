import pickle
import re
from pathlib import Path

import numpy as np
import napari


def load_ms_images(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    images = data["image"]
    meta = data["metadata"]

    # Stack into (window, cycle, mz)
    stack = np.stack(images, axis=0)

    # Build per-window metadata
    window_info = []

    # Invert DIA window index
    index_to_window = {0: ("MS1", None)}
    for (mz_lo, mz_hi), idx in meta["dia_windows"].items():
        index_to_window[idx] = ("MS2", (mz_lo, mz_hi))

    for i in range(stack.shape[0]):
        kind, mz = index_to_window[i]
        if kind == "MS1":
            info = {
                "window": i,
                "type": "MS1",
                "mz_range": (meta["start_mz"], meta["end_mz"]),
                "rt_range": (meta["start_rt"], meta["end_rt"]),
            }
        else:
            info = {
                "window": i,
                "type": "MS2",
                "mz_range": mz,
                "rt_range": (meta["start_rt"], meta["end_rt"]),
            }
        window_info.append(info)

    return stack, window_info


def compute_global_contrast(stack):
    # robust global contrast
    vmin = np.percentile(stack, 1)
    vmax = np.percentile(stack, 99)
    return vmin, vmax


def main(pkl_path):
    stack, window_info = load_ms_images(pkl_path)
    vmin, vmax = compute_global_contrast(stack)

    viewer = napari.Viewer(title="MS DIA Image Viewer")

    # Add stack as an image layer
    layer = viewer.add_image(
        stack,
        name="MS images",
        colormap="magma",
        contrast_limits=(vmin, vmax),
        scale=(1, 1, 1),  # (window, cycle, mz)
        blending="additive",
    )

    # Add a text overlay for metadata
    viewer.text_overlay.visible = True

    # Update text overlay when the **first dimension (window)** changes
    @viewer.dims.events.current_step.connect
    def update_text(event):
        window_idx = viewer.dims.current_step[0]  # first axis = window
        info = window_info[window_idx]

        viewer.text_overlay.text = (
            f"Window {info['window']} — {info['type']}\n"
            f"m/z: {info['mz_range'][0]:.1f}–{info['mz_range'][1]:.1f}\n"
            f"RT: {info['rt_range'][0]:.2f}–{info['rt_range'][1]:.2f} min"
        )

    napari.run()


def load_ms_images_cond(pkl_path,cond_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    with open(cond_path, "rb") as f:
        cond = pickle.load(f)

    images = data["image"]
    features = cond
    features = [np.zeros_like(features[0])]+features
    meta = data["metadata"]

    stack = np.stack(images, axis=0)
    feature_stack = np.stack(features, axis=0)

    window_info = []

    index_to_window = {0: ("MS1", None)}
    for (mz_lo, mz_hi), idx in meta["dia_windows"].items():
        index_to_window[idx] = ("MS2", (mz_lo, mz_hi))

    for i in range(stack.shape[0]):
        kind, mz = index_to_window[i]
        info = {
            "window": i,
            "type": kind,
            "mz_range": (meta["start_mz"], meta["end_mz"]) if kind == "MS1" else mz,
            "rt_range": (meta["start_rt"], meta["end_rt"]),
        }
        window_info.append(info)

    return stack, feature_stack, window_info


def compute_feature_contrast(feature_stack):
    vmin = np.percentile(feature_stack, 95)  # suppress background
    vmax = np.percentile(feature_stack, 99.9)
    return vmin, vmax


def _load_array(path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            array = pickle.load(f)
    else:
        import imageio.v3 as iio

        array = iio.imread(path)

    array = np.asarray(array)
    return np.squeeze(array)


def _slice_index(path):
    match = re.search(r"_ms2_(\d+)(?:_|\.|$)", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot find MS2 slice index in filename: {path}")
    return int(match.group(1))


def load_real_ms2_stack(real_pkl_path, expected_slices=100):
    with open(real_pkl_path, "rb") as f:
        data = pickle.load(f)

    images = data["image"] if isinstance(data, dict) else data
    stack = np.stack(images, axis=0)

    # The real file usually contains MS1 first, then the 100 MS2 windows.
    if expected_slices is not None and stack.shape[0] == expected_slices + 1:
        stack = stack[1:]

    if expected_slices is not None and stack.shape[0] != expected_slices:
        raise ValueError(
            f"Expected {expected_slices} real slices, found {stack.shape[0]} in {real_pkl_path}"
        )

    return stack


def load_generated_ms2_stack(generated_dir, extension=".pkl", expected_slices=100):
    generated_dir = Path(generated_dir)
    files = sorted(generated_dir.glob(f"*{extension}"), key=_slice_index)

    if expected_slices is not None and len(files) != expected_slices:
        raise ValueError(
            f"Expected {expected_slices} generated {extension} files, found {len(files)} in {generated_dir}"
        )

    indices = [_slice_index(path) for path in files]
    if expected_slices is not None and indices != list(range(expected_slices)):
        raise ValueError(f"Generated slice indices are not 0..{expected_slices - 1}: {indices}")

    return np.stack([_load_array(path) for path in files], axis=0), files


def _transform_real_stack_for_model(stack, im_size):
    from skimage.transform import resize

    transformed = []
    for image in stack:
        # Same slicing semantics as dataset.ms_dataset.CropTransform.
        image = image[90:90 + 422, 0:1024]
        image = resize(
            image,
            im_size,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )
        image = (image - 3.04) / 3.04
        transformed.append(image.astype(np.float32, copy=False))

    return np.stack(transformed, axis=0)


def compare_real_generated_images(
    real_pkl_path="data/results/real_img/KLEPNE-103-ANA_100vW_100SPD.pkl",
    generated_dir="data/results/old_gen",
    generated_extension=".pkl",
    expected_slices=100,
    im_size=None,
):
    real_stack = load_real_ms2_stack(real_pkl_path, expected_slices=expected_slices)
    generated_stack, generated_files = load_generated_ms2_stack(
        generated_dir,
        extension=generated_extension,
        expected_slices=expected_slices,
    )

    if im_size is None:
        im_size = generated_stack.shape[1:]

    real_transformed = _transform_real_stack_for_model(real_stack, im_size)

    if real_transformed.shape != generated_stack.shape:
        raise ValueError(
            f"Shape mismatch after real transform: real {real_transformed.shape}, "
            f"generated {generated_stack.shape}."
        )

    generated_stack = generated_stack.astype(np.float32, copy=False)
    diff_stack = np.abs(real_transformed - generated_stack)

    mse_per_slice = np.mean((real_transformed - generated_stack) ** 2, axis=(1, 2))
    mae_per_slice = np.mean(diff_stack, axis=(1, 2))

    viewer = napari.Viewer(title="Real vs Generated MS2 Comparison")
    viewer.add_image(
        real_transformed,
        name="real MS2 model transform",
        colormap="magma",
        blending="additive",
    )
    viewer.add_image(
        generated_stack,
        name="generated MS2",
        colormap="green",
        opacity=0.6,
        blending="additive",
    )
    viewer.add_image(
        diff_stack,
        name="absolute difference",
        colormap="cyan",
        opacity=0.8,
        visible=False,
        blending="additive",
    )

    viewer.text_overlay.visible = True

    @viewer.dims.events.current_step.connect
    def update_text(event):
        slice_idx = viewer.dims.current_step[0]
        viewer.text_overlay.text = (
            f"MS2 slice {slice_idx}\n"
            f"Generated: {generated_files[slice_idx].name}\n"
            f"MSE: {mse_per_slice[slice_idx]:.6f}\n"
            f"MAE: {mae_per_slice[slice_idx]:.6f}"
        )

    napari.run()

    return {
        "mse_per_slice": mse_per_slice,
        "mae_per_slice": mae_per_slice,
        "mean_mse": float(np.mean(mse_per_slice)),
        "mean_mae": float(np.mean(mae_per_slice)),
    }

def main_cond(pkl_path,cond_path):
    stack, feature_stack, window_info = load_ms_images_cond(pkl_path,cond_path)

    img_vmin, img_vmax = compute_global_contrast(stack)

    viewer = napari.Viewer(title="MS DIA Image Viewer")

    # Base MS image
    img_layer = viewer.add_image(
        stack,
        name="MS intensity",
        colormap="magma",
        contrast_limits=(img_vmin, img_vmax),
        blending="additive",
    )

    # Feature overlay
    # Amplify and normalize features to make them stand out
    feature_stack_scaled = feature_stack.copy()
    feature_stack_scaled *= 10  # amplify

    feat_vmin, feat_vmax = compute_feature_contrast(feature_stack_scaled)

    feature_layer = viewer.add_image(
        feature_stack_scaled,
        name="Detected peaks",
        colormap="blue",
        contrast_limits=(feat_vmin, feat_vmax),
        opacity=0.9,
        blending="additive",
        visible=True,
    )

    # Metadata overlay
    viewer.text_overlay.visible = True

    @viewer.dims.events.current_step.connect
    def update_text(event):
        window_idx = viewer.dims.current_step[0]
        info = window_info[window_idx]

        viewer.text_overlay.text = (
            f"Window {info['window']} — {info['type']}\n"
            f"m/z: {info['mz_range'][0]:.1f}–{info['mz_range'][1]:.1f}\n"
            f"RT: {info['rt_range'][0]:.2f}–{info['rt_range'][1]:.2f} min"
        )

    napari.run()


if __name__ == "__main__":
    import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python view_ms_images_napari.py images.pkl")
    #     sys.exit(1)

    # main('data/image_zeno/CITFRE-36-ANA_100vW_100SPD.pkl')
    # main_cond('data/test/KLEAER-20-AER-d200_mzml.pkl','data/test/KLEAER-20-AER-d200_conditioning_list_gaussian.pkl')
    compare_real_generated_images()

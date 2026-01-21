import pickle
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
    main_cond('data/test/KLEAER-20-AER-d200_mzml.pkl','data/test/KLEAER-20-AER-d200_conditioning_list_gaussian.pkl')



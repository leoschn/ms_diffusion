import glob
import pickle
import os
import numpy as np
import cv2

def inverse_transform_np(img,crop_top=90,crop_left=0,crop_h=422,crop_w=1024,original_shape=(512, 1024),mean=3.04,
                         std=3.04,final_size = (663, 901)):
    """
    img: numpy array (H, W)
    """

    # --- unnormalize ---
    img = img * std + mean

    # --- resize back to cropped size ---
    img = cv2.resize(img, (crop_w, crop_h))  # (W, H)

    # --- pad zeros to restore original shape ---
    H0, W0 = original_shape
    out = np.zeros((H0, W0), dtype=img.dtype)

    out[crop_top:crop_top+crop_h, crop_left:crop_left+crop_w] = img

    out = cv2.resize(out, (final_size[0], final_size[1]))
    return out


def reconstruct_image(base_name ,meta_data, base_size=(663,901),out_dir=None):
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)
    list_img = []
    n_wind = len(meta_data['dia_windows'])
    for i in range (n_wind):
        with open(base_name.replace('WIND_INDEX',str(i)), 'rb') as f:
            img =  pickle.load(f).squeeze()
            img = inverse_transform_np(img,final_size=base_size)
        list_img.append(img)
    data_out = {'image': list_img, 'metadata': meta_data}
    out_path = base_name.replace('WIND_INDEX','full')
    base_out_path  = os.path.basename(out_path)
    with open(os.path.join(out_dir,base_out_path), 'wb') as f:
        pickle.dump(data_out, f)

if __name__ == '__main__':
    full_name = glob.glob('/lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/sampled_full/**.pkl')
    DUMMY_IMAGE_PATH = '/lustre/fsn1/projects/rech/bun/ucg81ws/image/ESCCOL-246-ANA_100vW_100SPD.pkl'
    base_name_list = [filename.rsplit('_', 2)[0] for filename in full_name]
    base_name_list = list(set(base_name_list))
    common_meta_data = pickle.load(open(DUMMY_IMAGE_PATH, 'rb'))['metadata']
    final_size = pickle.load(open(DUMMY_IMAGE_PATH, 'rb'))['image'][0].shape
    for name in base_name_list:
        reconstruct_image(name + 'WIND_INDEX_9.pkl', meta_data=common_meta_data, base_size=final_size,
                          out_dir='/lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/sampled_full_reconstructed')
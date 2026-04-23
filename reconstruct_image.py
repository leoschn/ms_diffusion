import glob
import pickle
import os

def reconstruct_image(base_name ,meta_data,out_dir=None):
    #unpickle all 100 windows and concatenate it in a single .pkl file for latter use.
    if not(os.path.exists(out_dir)):
        os.mkdir(out_dir)
    list_img = []
    n_wind = len(meta_data['dia_windows'])
    for i in range (n_wind):
        with open(base_name.replace('WIND_INDEX',str(i)), 'rb') as f:
            img =  pickle.load(f).squeeze()
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
    for name in base_name_list:
        reconstruct_image(name + '_WIND_INDEX_9.pkl', meta_data=common_meta_data,
                          out_dir='/lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/sampled_full_reconstructed')
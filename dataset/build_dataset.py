import glob
import os
import pickle
import sys
from random import choice

from loess.loess_1d import loess_1d
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from findpeaks import findpeaks
from os import listdir
from os.path import isfile, join

from matplotlib import pyplot as plt


def load_lib(path):
    table = pq.read_table(path)
    table = table.to_pandas()

    return table


def cross_group(group_seq, group_lib):
    return group_seq.merge(group_lib, how='cross',suffixes=('_left', '_right'))



def extract_detected_features(sample_name,img_path,chimerys_report_path,diann_lib_path,fdr=0.05):
    diann_lib = load_lib(diann_lib_path)
    img_data = pickle.load(open(img_path, 'rb'))
    img_list = img_data['image']
    metadata = img_data['metadata']
    df_chimerys = pd.read_csv(chimerys_report_path, sep='\t')
    df_chimerys = df_chimerys[df_chimerys['PEP'] <= fdr]
    df_chimerys['MODIFIED_SEQUENCE'] = df_chimerys['MODIFIED_SEQUENCE'].apply(
        lambda x: x.replace('[UNIMOD:35]', '(UniMod:35)'))
    df_chimerys['X'] = ((df_chimerys['RETENTION_TIME'] - metadata['start_rt']) / metadata['span_rt']) * metadata[
        'max_cycle']
    df_chimerys['X'] = df_chimerys['X'].round(0).astype('int64')
    df_chimerys = df_chimerys[['X', 'SEQUENCE', 'PRECURSOR_CHARGE', 'SCAN_NUMBER_IN_FILE']].drop_duplicates()
    df_chimerys['SCAN_NUMBER_IN_FILE']=df_chimerys['SCAN_NUMBER_IN_FILE'] - 1
    seq_id = df_chimerys['SEQUENCE'].tolist()


    diann_lib['ms'] = (diann_lib['Precursor.Mz'] - metadata['list_precursor_mass_center'][0]) / 4 + 1
    diann_lib['ms'] = diann_lib['ms'].round().astype(int)
    diann_lib['Y'] = (((diann_lib['Product.Mz'] - metadata['ms1_start_mz']) / metadata['total_ms1_mz']) * metadata[
        'n_bin_ms1']).round(0).astype('int64')

    diann_lib_id = diann_lib[diann_lib['Modified.Sequence'].isin(seq_id)]

    print('nb_id total', len(seq_id))
    print('nb_frag total', len(diann_lib_id))
    nb_window = len(img_list)
    conditioning_list=[]
    for window in range(1, nb_window):
        image_window = img_list[window]
        conditioning = np.zeros_like(image_window)
        df_chimerys_window = df_chimerys[df_chimerys['SCAN_NUMBER_IN_FILE']% nb_window == window]

        df_combine_window = (
            df_chimerys_window
            .groupby(['SEQUENCE', 'PRECURSOR_CHARGE'], group_keys=False)
            .apply(lambda g: cross_group(g, diann_lib_id[
                (diann_lib_id['Modified.Sequence'] == g.name[0]) &
                (diann_lib_id['Precursor.Charge'] == g.name[1])
                ]))
            .reset_index(drop=True)
        )
        df_combine_window_valid = df_combine_window[(df_combine_window['X']< conditioning.shape[0]) & (df_combine_window['Y']< conditioning.shape[1])]
        for row in df_combine_window_valid.iterrows():
            Y = row[1]['Y']
            X = row[1]['X']
            if X < conditioning.shape[0] and Y < conditioning.shape[1]:
                conditioning[X, Y] += row[1]['Relative.Intensity']
        conditioning_list.append(conditioning)
        print(conditioning.sum())

    os.makedirs(f'data/conditioning_v2/{sample_name}',exist_ok=True)
    with open(f'data/conditioning_v2/{sample_name}/conditioning_list.pkl', 'wb') as f:
        pickle.dump(conditioning_list, f)

    return conditioning_list


def extract_detected_features_zeno(sample_name,img_path,diann_report,fdr=0.05):
    img_data = pickle.load(open(img_path, 'rb'))
    img_list = img_data['image']
    metadata = img_data['metadata']
    df_diann = pd.read_csv(diann_report, sep='\t')
    df_diann = df_diann[df_diann['PEP'] <= fdr]
    df_diann['X'] = ((df_diann['RT'] - metadata['start_rt']) / metadata['span_rt']) * metadata[
        'max_cycle'] #RT coordinate in the image
    df_diann['X'] = df_diann['X'].round(0).astype('int64')
    df_diann['scan_window'] = df_diann['MS2.Scan']%len(metadata['dia_windows']) #Vérifier qu'il n'y ait pas de décallage
    nb_window = len(img_list)
    conditioning_list=[]
    for window in range(1, nb_window):
        image_window = img_list[window]
        conditioning = np.zeros_like(image_window)
        df_window = df_diann[df_diann['scan_window'] == window]
        df_window_valid = df_window[(df_window['X']< conditioning.shape[0])]


        for row in df_window_valid.iterrows():
            X = row[1]['X']
            frag_info = row[1]['Fragment.Info'] #mz and name of frags
            frag_int = row[1]['Fragment.Quant.Raw']

            list_mz = frag_info.split(';')[:-1]
            list_mz = [i.split('/')[1] for i in list_mz] #mz of frags
            print(list_mz)
            list_mz = [
                int(round(
                    ((float(i) - metadata['start_mz']) /
                     (metadata['end_mz'] - metadata['start_mz'])) *
                    metadata['n_bin']
                ))
                for i in list_mz
            ]
            list_int = frag_int.split(';')

            for i in range(len(list_mz)):
                if list_mz[i] < conditioning.shape[1]:
                    conditioning[X, list_mz[i]] += list_int[i]

        conditioning_list.append(conditioning)
        print(conditioning.sum())

    os.makedirs(f'../data/test/{sample_name}',exist_ok=True)
    with open(f'../data/test/{sample_name}/conditioning_list.pkl', 'wb') as f:
        pickle.dump(conditioning_list, f)

    return conditioning_list



def extract_detected_features_zeno_2(sample_name, img_path, diann_report, fdr=0.05):
    import numpy as np
    import pandas as pd
    import pickle
    import os

    # ----------------------------
    # 1. Load image & metadata
    # ----------------------------
    img_data = pickle.load(open(img_path, 'rb'))
    img_list = img_data['image']
    metadata = img_data['metadata']

    # ----------------------------
    # 2. Load DIA-NN report
    # ----------------------------
    df_diann = pd.read_csv(diann_report, sep='\t')
    df_diann = df_diann[df_diann['PEP'] <= fdr]

    # ----------------------------
    # 3. Compute RT coordinate
    # ----------------------------
    df_diann['X'] = ((df_diann['RT'] - metadata['start_rt']) / metadata['span_rt']) * metadata['max_cycle']
    df_diann['X'] = df_diann['X'].round(0).astype('int64')

    # ----------------------------
    # 4. Map precursor.mz to DIA window index
    # ----------------------------
    def map_mz_to_window(prec_mz, dia_windows):
        """
        Returns the index of the DIA window that contains prec_mz.
        Uses metadata['dia_windows'] dictionary: {(low, high): index}
        """
        for (low, high), idx in dia_windows.items():
            if low <= prec_mz <= high:
                return idx
        return None  # outside any window

    df_diann['scan_window'] = df_diann['Precursor.Mz'].apply(lambda mz: map_mz_to_window(float(mz), metadata['dia_windows']))

    # Drop rows that could not be assigned to a window
    df_diann = df_diann.dropna(subset=['scan_window'])
    df_diann['scan_window'] = df_diann['scan_window'].astype('int')

    # ----------------------------
    # 5. Build conditioning images
    # ----------------------------
    nb_window = len(img_list)
    conditioning_list = []

    for window in range(1, nb_window):  # MS1 = 0
        image_window = img_list[window]
        conditioning = np.zeros_like(image_window)

        df_window = df_diann[df_diann['scan_window'] == window]
        df_window_valid = df_window[df_window['X'] < conditioning.shape[0]]

        for _, row in df_window_valid.iterrows():
            X = row['X']
            frag_info = row['Fragment.Info']  # mz/name
            frag_int = row['Fragment.Quant.Raw']

            list_mz = frag_info.split(';')[:-1]
            list_mz = [i.split('/')[1] for i in list_mz]  # get fragment mz
            mz = np.array(list_mz, dtype=np.float64)

            mz_idx = np.round(
                ((mz - metadata['start_mz']) /
                 (metadata['end_mz'] - metadata['start_mz'])) *
                metadata['n_bin']
            ).astype(np.int64)

            list_int = frag_int.split(';')[:-1]

            for i in range(len(list_mz)):
                if 0 <= mz_idx[i] < conditioning.shape[1]:
                    conditioning[X, mz_idx[i]] += float(list_int[i])

        conditioning_list.append(conditioning)
        print(f"Window {window}: total intensity = {conditioning.sum()}")

    # ----------------------------
    # 6. Save
    # ----------------------------
    out_dir = f'../data/test/{sample_name}'
    os.makedirs(out_dir, exist_ok=True)

    with open(f'{out_dir}/conditioning_list.pkl', 'wb') as f:
        pickle.dump(conditioning_list, f)

    return conditioning_list

def extract_detected_features_zeno_gaussian(out_path, img_path, diann_report, fdr=0.05):
    import numpy as np
    import pandas as pd
    import pickle
    import os

    # ----------------------------
    # 1. Load image & metadata
    # ----------------------------
    img_data = pickle.load(open(img_path, 'rb'))
    img_list = img_data['image']
    metadata = img_data['metadata']

    # ----------------------------
    # 2. Load DIA-NN report
    # ----------------------------
    df_diann = pd.read_csv(diann_report, sep='\t')
    df_diann = df_diann[df_diann['PEP'] <= fdr]

    # ----------------------------
    # 3. Compute RT coordinates (start / stop)
    # ----------------------------
    df_diann['X_start'] = (
                                  (df_diann['RT.Start'] - metadata['start_rt']) /
                                  metadata['span_rt']
                          ) * metadata['max_cycle']

    df_diann['X_stop'] = (
                                 (df_diann['RT.Stop'] - metadata['start_rt']) /
                                 metadata['span_rt']
                         ) * metadata['max_cycle']

    df_diann[['X_start', 'X_stop']] = (
        df_diann[['X_start', 'X_stop']]
        .round(0)
        .astype('int64')
    )

    # ----------------------------
    # 4. Map precursor.mz to DIA window index
    # ----------------------------
    def map_mz_to_window(prec_mz, dia_windows):
        """
        Returns the index of the DIA window that contains prec_mz.
        Uses metadata['dia_windows'] dictionary: {(low, high): index}
        """
        for (low, high), idx in dia_windows.items():
            if low <= prec_mz <= high:
                return idx
        return None  # outside any window

    df_diann['scan_window'] = df_diann['Precursor.Mz'].apply(lambda mz: map_mz_to_window(float(mz), metadata['dia_windows']))

    # Drop rows that could not be assigned to a window
    df_diann = df_diann.dropna(subset=['scan_window'])
    df_diann['scan_window'] = df_diann['scan_window'].astype('int')

    # ----------------------------
    # 5. Build conditioning images
    # ----------------------------
    nb_window = len(img_list)
    conditioning_list = []

    for window in range(1, nb_window):  # MS1 = 0
        image_window = img_list[window]
        conditioning = np.zeros_like(image_window)

        df_window = df_diann[df_diann['scan_window'] == window]
        df_window_valid = df_window[
            (df_window['X_start'] < conditioning.shape[0]) &
            (df_window['X_stop'] >= 0)
            ]

        for _, row in df_window_valid.iterrows():
            X0 = max(0, row['X_start'])
            X1 = min(conditioning.shape[0] - 1, row['X_stop'])
            if X1 < X0:
                continue

            mu = 0.5 * (X0 + X1)
            width = max(1, X1 - X0 + 1)
            sigma = width / 6.0

            rt_axis = np.arange(X0, X1 + 1)
            gauss = np.exp(-0.5 * ((rt_axis - mu) / sigma) ** 2)
            gauss /= gauss.sum()

            frag_info = row['Fragment.Info']
            frag_int = row['Fragment.Quant.Raw']

            list_mz = [i.split('/')[1] for i in frag_info.split(';')[:-1]]
            mz = np.array(list_mz, dtype=np.float64)

            mz_idx = np.round(
                ((mz - metadata['start_mz']) /
                 (metadata['end_mz'] - metadata['start_mz'])) *
                metadata['n_bin']
            ).astype(np.int64)

            list_int = frag_int.split(';')[:-1]

            for i in range(len(mz_idx)):
                if 0 <= mz_idx[i] < conditioning.shape[1]:
                    conditioning[X0:X1 + 1, mz_idx[i]] += float(list_int[i]) * gauss

        conditioning_list.append(conditioning)
        print(f"Window {window}: total intensity = {conditioning.sum()}")

    # ----------------------------
    # 6. Save
    # ----------------------------

    with open(out_path, 'wb') as f:
        pickle.dump(conditioning_list, f)

    return conditioning_list


def build_training_pairs(img_path,cond_path,out_dir,sample_name):
    img_data = pickle.load(open(img_path, 'rb'))['image']
    cond_data = pickle.load(open(cond_path, 'rb'))
    assert len(img_data) == len(cond_data)+1
    os.makedirs(out_dir,exist_ok=True)
    for window in range(len(cond_data)):
        img = img_data[window]+1
        cond = cond_data[window]
        with open(os.path.join(out_dir,f'{sample_name}_ms2_{window}.pkl'), 'wb') as f:
            pickle.dump((img,cond), f)


def main():
    # with open(sys.argv[1], 'r') as f:
    #     sample_name = f.readline()
    #     if not (os.path.exists(f'/lustre/fsn1/projects/rech/bun/ucg81ws/conditioning/conditioning_{sample_name}.pkl')):
    #         extract_detected_features_zeno_gaussian(
    #             out_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/conditioning/conditioning_{sample_name}.pkl',
    #             img_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{sample_name}.pkl',
    #             diann_report=f'/lustre/fsn1/projects/rech/bun/ucg81ws/output/report_{sample_name}.tsv')
    #         build_training_pairs(img_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{sample_name}.pkl',
    #             cond_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/conditioning/conditioning_{sample_name}.pkl',
    #             out_dir='/lustre/fsn1/projects/rech/bun/ucg81ws/pairs/',
    #             sample_name=sample_name)

    sample_list = glob.glob('/lustre/fsn1/projects/rech/bun/ucg81ws/mzml/**.mzML', recursive=True)
    for f_name in sample_list:
        # if not os.path.exists(f'/lustre/fsn1/projects/rech/bun/ucg81ws/conditioning/conditioning_{sample_name}.pkl'):
        sample_name = os.path.basename(f_name).split('.mzML')[0]
        #     extract_detected_features_zeno_gaussian(out_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/conditioning/conditioning_{sample_name}.pkl',
        #                                             img_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{sample_name}.pkl',
        #                                             diann_report=f'/lustre/fsn1/projects/rech/bun/ucg81ws/output/report_{sample_name}.tsv')
        build_training_pairs(img_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{sample_name}.pkl',
                             cond_path=f'/lustre/fsn1/projects/rech/bun/ucg81ws/conditioning/conditioning_{sample_name}.pkl',
                             out_dir='/lustre/fsn1/projects/rech/bun/ucg81ws/pairs/',
                             sample_name=sample_name)


if __name__ == '__main__':
    main()
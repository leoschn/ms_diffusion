import os
import pickle
import re
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

def build_training_pairs(l,out_dir):
    for sample in l:
        img_data = pickle.load(open(f'data/image/{sample}.pkl', 'rb'))['image']
        cond_data = pickle.load(open(f'data/conditioning_v2/{sample}/conditioning_list.pkl', 'rb'))
        assert len(img_data) == len(cond_data)+1
        os.makedirs(out_dir,exist_ok=True)
        for window in range(1,len(cond_data)+1):
            img = img_data[window]
            cond = cond_data[window-1]
            with open(os.path.join(out_dir,f'{sample}_ms2_{window}.pkl', 'wb')) as f:
                pickle.dump((img,cond), f)

def plot_random_pairs():
    mypath = '../data/processed_pairs'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    f_name = choice(onlyfiles)
    with open(f'../data/processed_pairs/{f_name}', 'rb') as f:
        img,cond = pickle.load(f)
    fig, axs = plt.subplots(2)
    axs[0].imshow(img)
    axs[1].imshow(cond)
    plt.savefig(f'..img_{f_name}.png')



def main():
    #
    # for sample in ['ESCCOL100','CANGLA10','KLEPNE164_hemoc','PSEAER286','STAHOM8_AER','CITFRE65','ESCCOL121','KLEPNE172','STAAU36','STAHOM8_ANA','ACIBAU130','ENCFAC56','ESCCOL259','KLEPNE86','STAAU81','STCPNE10','ENTCLO18','KLEOXY23','PSEAER154','STAEPI11_AER','STCPYO20','CANALB32','ENTHOR84','KLEPNE164_bdg','PSEAER259','STAEPI11_ANA']:
    #     print(sample)
    #     specie = re.split(r'(?=\d)', sample)[0]
    #     cond_list = extract_detected_features(sample_name=sample,img_path=f'data/image/{sample}.pkl',
    #                                           chimerys_report_path=f'data/chimerys/{sample}/psms.tsv',
    #                                           diann_lib_path=f'data/library/{specie}_universal_cont_blood.parquet',
    #                                           fdr=0.05)

    build_training_pairs(['ACIBAU130','CANALB32'],'data/processed_pairs_v2/train')
    build_training_pairs(['ESCCOL100', 'CANGLA10', 'KLEPNE164_hemoc', 'PSEAER286', 'STAHOM8_AER', 'CITFRE65', 'ESCCOL121', 'KLEPNE172',
     'STAAU36', 'STAHOM8_ANA',  'ENCFAC56', 'ESCCOL259', 'KLEPNE86', 'STAAU81', 'STCPNE10', 'ENTCLO18',
     'KLEOXY23', 'PSEAER154', 'STAEPI11_AER', 'STCPYO20', 'ENTHOR84', 'KLEPNE164_bdg', 'PSEAER259',
     'STAEPI11_ANA'],'data/processed_pairs_v2/test')
if __name__ == '__main__':
    main()
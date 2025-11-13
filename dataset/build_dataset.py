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



def extract_detected_features(sample_name,img_path,chimerys_report_path,diann_lib_path,ref_align_path,aligned=False,matching_option=1,fdr=0.05):
    img_data=pickle.load(open(img_path, 'rb'))
    img_list = img_data['image']
    metadata = img_data['metadata']
    df_chimerys = pd.read_csv(chimerys_report_path,sep='\t')
    # df_chimerys['RETENTION_TIME'] = (df_chimerys['MIN_RETENTION_TIME'] + df_chimerys['MAX_RETENTION_TIME']) / 2
    df_chimerys = df_chimerys[df_chimerys['PEP']<=fdr]
    df_chimerys['MODIFIED_SEQUENCE'] = df_chimerys['MODIFIED_SEQUENCE'].apply(
        lambda x: x.replace('[UNIMOD:35]', '(UniMod:35)'))
    df_chimerys['X']=((df_chimerys['RETENTION_TIME'] - metadata['start_rt']) / metadata['span_rt']) * metadata['max_cycle']
    df_chimerys['X']=df_chimerys['X'].round(0).astype('int64')
    df_chimerys=df_chimerys[['X','SEQUENCE','PRECURSOR_CHARGE','SCAN_NUMBER_IN_FILE']].drop_duplicates()
     #TODO charge

    diann_lib = load_lib(diann_lib_path)


    #initialise peak detection
    if matching_option == 2:
        fp = findpeaks(method='topology', whitelist=['peak'], imsize=(img_list[0].shape[1], img_list[0].shape[0]),
                       limit=20, denoise=None, verbose='off', scale=True)

    #align RT
    print('aligning RT')


    if not aligned:
        ref_align = pd.read_csv(ref_align_path, sep='\t')
        ref_align['RT_expe'] = (ref_align['MIN_RETENTION_TIME'] + ref_align['MAX_RETENTION_TIME']) / 2

        df_ref_align = load_lib('data/library/lib_candida_albicans.parquet')
        df_ref_align['SEQUENCE'] = df_ref_align['Stripped.Sequence']

        df_tuning_rt = ref_align.join(df_ref_align.set_index('SEQUENCE'), on='SEQUENCE', how='inner')

        xout, yout, wout = loess_1d(np.array(df_tuning_rt['RT'].tolist()), np.array(df_tuning_rt['RT_expe'].tolist()),
                                    xnew=diann_lib['RT'],
                                    degree=1,
                                    npoints=None, rotate=False, sigy=None)

        diann_lib['Aligned_RT'] = yout
        specie = re.split(r'(?=\d)', sample_name)[0]
        diann_lib.to_parquet(f'data/library/{specie}_aligned.parquet')
    diann_lib = diann_lib[['Modified.Sequence','Aligned_RT','RT','Precursor.Mz','Product.Mz','Precursor.Charge','Fragment.Type','Fragment.Series.Number','Relative.Intensity']]
    # diann_lib = diann_lib[diann_lib['Modified.Sequence'].map(lambda x: x in identified_seq)]

    diann_lib['ms'] = (diann_lib['Precursor.Mz']-metadata['list_precursor_mass_center'][0])/4 +1
    diann_lib['ms'] = diann_lib['ms'].round().astype(int)
    diann_lib['X'] = (((diann_lib['Aligned_RT'] - metadata['start_rt']) / metadata['span_rt']) * metadata['max_cycle']).round(0).astype('int64')
    diann_lib['Y'] = (((diann_lib['Product.Mz'] - metadata['ms1_start_mz']) / metadata['total_ms1_mz']) * metadata['n_bin_ms1']).round(0).astype('int64')


    # for each identified peptide use diann to predict theoretical spectra :
    # option 1 => use these spectra to directly compute conditioning image (with relative intensity)
    # option 2 => use these spectra to match peak on the experimental image and build conditioning image based on matched peak and their measured relative intensity. Matching can be utterly restrictive (no RT shift + no mz shift) (exact correspondence)

    conditioning_list=[]
    #option 1 use all predicted peak from detected peptides as conditioning
    if matching_option == 1:
        seq_id = df_chimerys['SEQUENCE'].tolist()
        diann_lib_id = diann_lib[diann_lib['Modified.Sequence'].isin(seq_id)]
        print('nb_id total',len(seq_id))
        print('nb_frag total',len(diann_lib_id))
        nb_window = len(img_list)
        for window in range(1,nb_window):
            image_window = img_list[window]
            conditioning=np.zeros_like(image_window)
            df_chimerys_window = df_chimerys[df_chimerys['SCAN_NUMBER_IN_FILE'] % nb_window==window]

            df_combine_window = (
                df_chimerys_window
                .groupby(['SEQUENCE', 'PRECURSOR_CHARGE'], group_keys=False)
                .apply(lambda g: cross_group(g, diann_lib_id[
                    (diann_lib_id['Modified.Sequence'] == g.name[0]) &
                    (diann_lib_id['Precursor.Charge'] == g.name[1])
                    ]))
                .reset_index(drop=True)
            )

            print(df_combine_window.shape)
            for row in  df_combine_window.iterrows():
                Y = row[1]['Y']
                X = row[1]['X_left']
                if X < conditioning.shape[0] and Y < conditioning.shape[1]:
                    conditioning[X,Y]+=row[1]['Relative.Intensity']
            conditioning_list.append(conditioning)
            print(conditioning.sum())



    #option 2 use predicted peak from detected peptides as conditioning ONLY IF they have been also detected
    if matching_option == 2:
        seq_id = df_chimerys['SEQUENCE'].tolist()
        diann_lib_id = diann_lib[diann_lib['Modified.Sequence'].isin(seq_id)]
        nb_window = len(img_list)
        for window in range(1,nb_window):
            image_window = img_list[window]
            conditioning=np.zeros_like(image_window)
            res = fp.fit(img_list[window])
            df_chimerys_window = df_chimerys[df_chimerys['SCAN_NUMBER_IN_FILE'] % nb_window==window]
            df_detected_peaks = pd.DataFrame(res['persistence'])

            df_combine_window = (
                df_chimerys_window
                .groupby(['SEQUENCE', 'PRECURSOR_CHARGE'], group_keys=False)
                .apply(lambda g: cross_group(g, diann_lib_id[
                    (diann_lib_id['Modified.Sequence'] == g.name[0]) &
                    (diann_lib_id['Precursor.Charge'] == g.name[1])
                    ]))
                .reset_index(drop=True)
            )

            df_revelant_peaks = pd.merge(df_combine_window,df_detected_peaks,left_on=['X','Y'],right_on=['X','Y'],how='inner')

            for row in df_revelant_peaks.iterrows():
                if row[1]['X'] < conditioning.shape[0] and row[1]['Y'] < conditioning.shape[1]:
                    conditioning[row[1]['X'], row[1]['Y']] += row[1]['Relative.Intensity']
            conditioning_list.append(conditioning)

    os.makedirs(f'data/conditioning/{sample_name}',exist_ok=True)
    pickle.dump(conditioning_list, open(f'data/conditioning/{sample_name}/conditioning_list.pkl', 'wb'))

    return conditioning_list

def build_training_pairs():
    for sample in ['ESCCOL100','CANGLA10','KLEPNE164_hemoc','PSEAER286','STAHOM8_AER','CITFRE65','ESCCOL121','KLEPNE172','STAAU36','STAHOM8_ANA','ACIBAU130','ENCFAC56','ESCCOL259','KLEPNE86','STAAU81','STCPNE10','ENTCLO18','KLEOXY23','PSEAER154','STAEPI11_AER','STCPYO20','CANALB32','ENTHOR84','KLEPNE164_bdg','PSEAER259','STAEPI11_ANA']:
        img_data = pickle.load(open(f'../data/image/{sample}.pkl', 'rb'))['image']
        cond_data = pickle.load(open(f'../data/conditioning/{sample}/conditioning_list.pkl', 'rb'))
        for window in range(1,len(img_data)):
            img = img_data[window]
            cond = cond_data[window-1]
            with open(f'../data/processed_pairs/{sample}_ms2_{window}.pkl', 'wb') as f:
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
    for sample in ['ESCCOL100','CANGLA10','KLEPNE164_hemoc','PSEAER286','STAHOM8_AER','CITFRE65','ESCCOL121','KLEPNE172','STAAU36','STAHOM8_ANA','ACIBAU130','ENCFAC56','ESCCOL259','KLEPNE86','STAAU81','STCPNE10','ENTCLO18','KLEOXY23','PSEAER154','STAEPI11_AER','STCPYO20','CANALB32','ENTHOR84','KLEPNE164_bdg','PSEAER259','STAEPI11_ANA']:
        print(sample)
        specie = re.split(r'(?=\d)', sample)[0]
        cond_list = extract_detected_features(sample_name=sample,img_path=f'data/image/{sample}.pkl',
                                              chimerys_report_path=f'data/chimerys/{sample}/psms.tsv',
                                              diann_lib_path=f'data/library/{specie}_universal_cont_blood.parquet',
                                              ref_align_path=f'data/chimerys/alignment/precursors.tsv', aligned=False,
                                              matching_option=1,fdr=0.05)

if __name__ == '__main__':
    main()

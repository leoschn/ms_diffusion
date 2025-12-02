import glob
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd


# file_names = glob.glob(os.path.join('data/processed_pairs/train/', '*'))
# n=0
# z=0
# for file_name in file_names:
#     n+=1
#     with open(file_name, 'rb') as f:
#         data = pickle.load(f)

    # name = os.path.basename(file_name).replace('.pkl', '')
    # print(name)
    # sample = name.split('_')[0]
    # scan = name.split('_')[2]
    # plt.imshow(data[0])
    # plt.savefig(f'image2/{scan}_ms2_{sample}.png')
    # plt.clf()
    # if data[1].sum()>1 :
    #     z+=1


with open('data/processed_pairs/test/ACIBAU130_ms2_2.pkl', 'rb') as f:
    data = pickle.load(f)
    plt.imshow(data[0])
    plt.savefig('test.png')
    plt.clf()
    plt.imshow(data[1])
    plt.savefig('test_cond.png')

    # 60% sans aucun pick detecté
    # front de MS1 compliquer a prédire sans info dur le scan number
    # partie hors analyse a enlever (présente partout cf netoyage ??)


def compute_prop_cond(path):
    df = pd.read_csv(path,sep='\t')
    df = df[df['PEP']<0.01]
    df['scan_mod']=df['SCAN_NUMBER_IN_FILE']%164
    scan_set=set(df['scan_mod'])
    return df,scan_set

# df,scan_set = compute_prop_cond('data/chimerys/ENCFAC56/psms.tsv')
# print(len(scan_set))
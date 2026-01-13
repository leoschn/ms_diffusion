import glob
import os

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pyRawMSDataReader.pyRawMSDataReader.RawFileReader import RawFileReader
from pyRawMSDataReader.pyRawMSDataReader.WiffFileReader_py import WiffFileReader
import numpy as np
import pickle




def build_image_ms2_raw(path_raw,out_path,bin_mz=1):
    # load raw data
    rawFile = RawFileReader(path_raw)
    max_cycle = 0

    start_rt = rawFile.GetStartTime()
    end_rt = rawFile.GetEndTime()
    span_rt = end_rt - start_rt

    first_scan, last_scan = rawFile.GetFirstSpectrumNumber(), rawFile.GetLastSpectrumNumber()
    list_precursor_mass_center=[]
    for scanNumber in range(first_scan, last_scan):
        if rawFile.GetMSOrderForScanNum(scanNumber).ToString() == 'Ms':
            ms1_start_mz = rawFile.LowMass
            ms1_end_mz = rawFile.HighMass
            max_cycle += 1
        elif rawFile.GetPrecursorMassForScanNum(scanNumber) not in list_precursor_mass_center:
            list_precursor_mass_center.append(rawFile.GetPrecursorMassForScanNum(scanNumber))

    print('start', ms1_start_mz, 'end', ms1_end_mz)
    total_ms1_mz = ms1_end_mz - ms1_start_mz

    n_bin_ms1 = int(total_ms1_mz // bin_mz)
    size_bin_ms1 = total_ms1_mz / n_bin_ms1
    list_img = [np.zeros([max_cycle, n_bin_ms1+1]) for i in range(len(list_precursor_mass_center)+1)]
    cycle = 0
    dict_int={}
    ind=1

    for mass in list_precursor_mass_center:
        dict_int[mass] = ind
        ind+=1

    for scanNumber in range(first_scan, last_scan):
        masses, intensities = rawFile.GetCentroidMassListFromScanNum(scanNumber)
        line = np.zeros(n_bin_ms1+1)
        if len(masses) > 0:
            for k in range(len(masses)):
                line[int((masses[k] - ms1_start_mz) // size_bin_ms1)] += intensities[k]
        if rawFile.GetMSOrderForScanNum(scanNumber).ToString() == 'Ms':
            list_img[0][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin_ms1+1))
        else :
            ind = dict_int[rawFile.GetPrecursorMassForScanNum(scanNumber)]
            list_img[ind][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin_ms1+1))
            if rawFile.GetPrecursorMassForScanNum(scanNumber) == list_precursor_mass_center[-1]:
                cycle += 1

    meta_data = {'n_bin_ms1': n_bin_ms1, 'size_bin_ms1': size_bin_ms1, 'ms1_start_mz' :ms1_start_mz,
            'ms1_end_mz' : ms1_end_mz,'max_cycle' : max_cycle,'list_precursor_mass_center':list_precursor_mass_center,
            'total_ms1_mz':total_ms1_mz,'start_rt':start_rt,'end_rt':end_rt,'span_rt':span_rt}

    data_out = {'image': list_img,'metadata':meta_data}
    if out_path is not None :
        with open(out_path, 'wb') as f:
            print('saving images to', out_path)
            pickle.dump(data_out, f)
    return data_out


def build_image_ms2_wiff(path_wiff,out_path,bin_mz=1):
    # load raw data
    wiffFile = WiffFileReader(path_wiff)
    max_cycle = 0

    start_rt = wiffFile.GetStartTime()
    end_rt = wiffFile.GetEndTime()
    span_rt = end_rt - start_rt

    first_scan, last_scan = wiffFile.GetFirstSpectrumNumber(), wiffFile.GetLastSpectrumNumber()
    list_precursor_mass_center=[]
    for scanNumber in range(first_scan, last_scan):
        if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
            ms1_start_mz = wiffFile.source.ScanInfos[scanNumber].LowMz
            ms1_end_mz = wiffFile.source.ScanInfos[scanNumber].HighMz
            print('ms1 ',wiffFile.source.ScanInfos[scanNumber].LowMz)
            print('ms1 ',wiffFile.source.ScanInfos[scanNumber].HighMz)
            max_cycle += 1
        elif wiffFile.GetPrecursorMassForScanNum(scanNumber) not in list_precursor_mass_center:
            list_precursor_mass_center.append(wiffFile.GetPrecursorMassForScanNum(scanNumber))
    list_precursor_mass_center.sort()

    start_mz = ms1_start_mz
    end_mz  = ms1_end_mz
    print('start', ms1_start_mz, 'end', ms1_end_mz)
    total_mz = end_mz - start_mz

    n_bin = int(total_mz // bin_mz)
    size_bin = total_mz / n_bin
    list_img = [np.zeros([max_cycle, n_bin+1]) for i in range(len(list_precursor_mass_center)+1)]
    cycle = 0
    dict_int={}
    ind=1

    for mass in list_precursor_mass_center:
        dict_int[mass] = ind
        ind+=1

    for scanNumber in range(first_scan, last_scan):
        masses, intensities = wiffFile.GetProfileMassListFromScanNum(scanNumber)
        line = np.zeros(n_bin+1)
        if len(masses) > 0:
            for k in range(len(masses)):
                if masses[k] < end_mz and masses[k] > start_mz:
                    line[int((masses[k] - ms1_start_mz) // size_bin)] += intensities[k]
        if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
            list_img[0][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin+1))
        else :
            ind = dict_int[wiffFile.GetPrecursorMassForScanNum(scanNumber)]
            list_img[ind][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin+1))
            if wiffFile.GetPrecursorMassForScanNum(scanNumber) == list_precursor_mass_center[-1]:
                cycle += 1

    meta_data = {'n_bin_ms1': n_bin, 'size_bin_ms1': size_bin, 'ms1_start_mz' :start_mz,
            'ms1_end_mz' : end_mz,'max_cycle' : max_cycle,'list_precursor_mass_center':list_precursor_mass_center,
            'total_ms1_mz':total_mz,'start_rt':start_rt,'end_rt':end_rt,'span_rt':span_rt}

    data_out = {'image': list_img,'metadata':meta_data}
    if out_path is not None :
        with open(out_path, 'wb') as f:
            print('saving images to', out_path)
            pickle.dump(data_out, f)
    wiffFile.Close()
    return data_out

def build_image_ms2_wiff_2(path_wiff, out_path, bin_mz=1):
    import numpy as np
    import pickle
    from collections import OrderedDict

    wiffFile = WiffFileReader(path_wiff)

    start_rt = wiffFile.GetStartTime()
    end_rt = wiffFile.GetEndTime()
    span_rt = end_rt - start_rt

    first_scan = wiffFile.GetFirstSpectrumNumber()
    last_scan = wiffFile.GetLastSpectrumNumber()

    # ----------------------------
    # 1. Identify MS1 range and DIA windows
    # ----------------------------
    dia_window_set = set()
    max_cycle = 0
    ms1_start_mz = None
    ms1_end_mz = None

    for scanNumber in range(first_scan, last_scan):
        scan = wiffFile.source.ScanInfos[scanNumber]

        if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
            ms1_start_mz = scan.LowMz
            ms1_end_mz = scan.HighMz
            max_cycle += 1
        else:
            dia_window_set.add((scan.LowMz, scan.HighMz)) # MS1 is index 0
    dia_windows = sorted(dia_window_set, key=lambda x: (x[0], x[1]))
    window_to_index = {
        win: i + 1  # MS1 = 0
        for i, win in enumerate(dia_windows)
    }
    start_mz = ms1_start_mz
    end_mz = ms1_end_mz
    total_mz = end_mz - start_mz

    n_bin = int(total_mz // bin_mz)
    size_bin = total_mz / n_bin

    # ----------------------------
    # 2. Allocate images
    # ----------------------------
    list_img = [
        np.zeros((max_cycle, n_bin + 1), dtype=np.float32)
        for _ in range(len(dia_windows) + 1)
    ]

    # ----------------------------
    # 3. Fill images
    # ----------------------------
    cycle = -1  # will increment on MS1

    for scanNumber in range(first_scan, last_scan):
        masses, intensities = wiffFile.GetProfileMassListFromScanNum(scanNumber)
        scan = wiffFile.source.ScanInfos[scanNumber]

        # increment cycle on MS1
        if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
            cycle += 1

        if cycle < 0 or len(masses) == 0:
            continue

        # vectorized binning (same bins for MS1 and MS2)
        bins = ((masses - start_mz) / size_bin).astype(int)
        valid = (bins >= 0) & (bins <= n_bin)

        line = np.zeros(n_bin + 1, dtype=np.float32)
        np.add.at(line, bins[valid], intensities[valid])

        # safe log transform
        line = np.log10(line + 1)

        if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
            list_img[0][cycle, :] = line
        else:
            key = (scan.LowMz, scan.HighMz)
            ind = window_to_index[key]
            list_img[ind][cycle, :] = line

    # ----------------------------
    # 4. Metadata
    # ----------------------------
    meta_data = {
        'n_bin': n_bin,
        'size_bin': size_bin,
        'start_mz': start_mz,
        'end_mz': end_mz,
        'max_cycle': max_cycle,
        'dia_windows': window_to_index,
        'start_rt': start_rt,
        'end_rt': end_rt,
        'span_rt': span_rt
    }

    data_out = {'image': list_img, 'metadata': meta_data}

    if out_path is not None:
        with open(out_path, 'wb') as f:
            print('saving images to', out_path)
            pickle.dump(data_out, f)

    wiffFile.Close()
    return data_out

if __name__ == '__main__':
    print('building image')
    sample_list = glob.glob('/lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/data/wiff_zeno/**.wiff', recursive=True)
    print(sample_list)
    for sample in sample_list:
        f_name = os.path.basename(sample).split('.wiff')[0]
        print(f_name)
        data_out = build_image_ms2_wiff_2(f'//lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/data/wiff_zeno/{f_name}.wiff',f'//lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/data/image_zeno/{f_name}.pkl')
        break

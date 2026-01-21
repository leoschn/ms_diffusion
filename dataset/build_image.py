import glob
import os

import sys
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# from pyRawMSDataReader.pyRawMSDataReader.RawFileReader import RawFileReader
# from pyRawMSDataReader.pyRawMSDataReader.WiffFileReader_py import WiffFileReader
import numpy as np
import pickle



#
# def build_image_ms2_raw(path_raw,out_path,bin_mz=1):
#     # load raw data
#     rawFile = RawFileReader(path_raw)
#     max_cycle = 0
#
#     start_rt = rawFile.GetStartTime()
#     end_rt = rawFile.GetEndTime()
#     span_rt = end_rt - start_rt
#
#     first_scan, last_scan = rawFile.GetFirstSpectrumNumber(), rawFile.GetLastSpectrumNumber()
#     list_precursor_mass_center=[]
#     for scanNumber in range(first_scan, last_scan):
#         if rawFile.GetMSOrderForScanNum(scanNumber).ToString() == 'Ms':
#             ms1_start_mz = rawFile.LowMass
#             ms1_end_mz = rawFile.HighMass
#             max_cycle += 1
#         elif rawFile.GetPrecursorMassForScanNum(scanNumber) not in list_precursor_mass_center:
#             list_precursor_mass_center.append(rawFile.GetPrecursorMassForScanNum(scanNumber))
#
#     print('start', ms1_start_mz, 'end', ms1_end_mz)
#     total_ms1_mz = ms1_end_mz - ms1_start_mz
#
#     n_bin_ms1 = int(total_ms1_mz // bin_mz)
#     size_bin_ms1 = total_ms1_mz / n_bin_ms1
#     list_img = [np.zeros([max_cycle, n_bin_ms1+1]) for i in range(len(list_precursor_mass_center)+1)]
#     cycle = 0
#     dict_int={}
#     ind=1
#
#     for mass in list_precursor_mass_center:
#         dict_int[mass] = ind
#         ind+=1
#
#     for scanNumber in range(first_scan, last_scan):
#         masses, intensities = rawFile.GetCentroidMassListFromScanNum(scanNumber)
#         line = np.zeros(n_bin_ms1+1)
#         if len(masses) > 0:
#             for k in range(len(masses)):
#                 line[int((masses[k] - ms1_start_mz) // size_bin_ms1)] += intensities[k]
#         if rawFile.GetMSOrderForScanNum(scanNumber).ToString() == 'Ms':
#             list_img[0][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin_ms1+1))
#         else :
#             ind = dict_int[rawFile.GetPrecursorMassForScanNum(scanNumber)]
#             list_img[ind][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin_ms1+1))
#             if rawFile.GetPrecursorMassForScanNum(scanNumber) == list_precursor_mass_center[-1]:
#                 cycle += 1
#
#     meta_data = {'n_bin_ms1': n_bin_ms1, 'size_bin_ms1': size_bin_ms1, 'ms1_start_mz' :ms1_start_mz,
#             'ms1_end_mz' : ms1_end_mz,'max_cycle' : max_cycle,'list_precursor_mass_center':list_precursor_mass_center,
#             'total_ms1_mz':total_ms1_mz,'start_rt':start_rt,'end_rt':end_rt,'span_rt':span_rt}
#
#     data_out = {'image': list_img,'metadata':meta_data}
#     if out_path is not None :
#         with open(out_path, 'wb') as f:
#             print('saving images to', out_path)
#             pickle.dump(data_out, f)
#     return data_out
#
#
# def build_image_ms2_wiff(path_wiff,out_path,bin_mz=1):
#     # load raw data
#     wiffFile = WiffFileReader(path_wiff)
#     max_cycle = 0
#
#     start_rt = wiffFile.GetStartTime()
#     end_rt = wiffFile.GetEndTime()
#     span_rt = end_rt - start_rt
#
#     first_scan, last_scan = wiffFile.GetFirstSpectrumNumber(), wiffFile.GetLastSpectrumNumber()
#     list_precursor_mass_center=[]
#     for scanNumber in range(first_scan, last_scan):
#         if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
#             ms1_start_mz = wiffFile.source.ScanInfos[scanNumber].LowMz
#             ms1_end_mz = wiffFile.source.ScanInfos[scanNumber].HighMz
#             print('ms1 ',wiffFile.source.ScanInfos[scanNumber].LowMz)
#             print('ms1 ',wiffFile.source.ScanInfos[scanNumber].HighMz)
#             max_cycle += 1
#         elif wiffFile.GetPrecursorMassForScanNum(scanNumber) not in list_precursor_mass_center:
#             list_precursor_mass_center.append(wiffFile.GetPrecursorMassForScanNum(scanNumber))
#     list_precursor_mass_center.sort()
#
#     start_mz = ms1_start_mz
#     end_mz  = ms1_end_mz
#     print('start', ms1_start_mz, 'end', ms1_end_mz)
#     total_mz = end_mz - start_mz
#
#     n_bin = int(total_mz // bin_mz)
#     size_bin = total_mz / n_bin
#     list_img = [np.zeros([max_cycle, n_bin+1]) for i in range(len(list_precursor_mass_center)+1)]
#     cycle = 0
#     dict_int={}
#     ind=1
#
#     for mass in list_precursor_mass_center:
#         dict_int[mass] = ind
#         ind+=1
#
#     for scanNumber in range(first_scan, last_scan):
#         masses, intensities = wiffFile.GetProfileMassListFromScanNum(scanNumber)
#         line = np.zeros(n_bin+1)
#         if len(masses) > 0:
#             for k in range(len(masses)):
#                 if masses[k] < end_mz and masses[k] > start_mz:
#                     line[int((masses[k] - ms1_start_mz) // size_bin)] += intensities[k]
#         if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
#             list_img[0][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin+1))
#         else :
#             ind = dict_int[wiffFile.GetPrecursorMassForScanNum(scanNumber)]
#             list_img[ind][cycle, :] = np.maximum(np.log10(line), np.zeros(n_bin+1))
#             if wiffFile.GetPrecursorMassForScanNum(scanNumber) == list_precursor_mass_center[-1]:
#                 cycle += 1
#
#     meta_data = {'n_bin_ms1': n_bin, 'size_bin_ms1': size_bin, 'ms1_start_mz' :start_mz,
#             'ms1_end_mz' : end_mz,'max_cycle' : max_cycle,'list_precursor_mass_center':list_precursor_mass_center,
#             'total_ms1_mz':total_mz,'start_rt':start_rt,'end_rt':end_rt,'span_rt':span_rt}
#
#     data_out = {'image': list_img,'metadata':meta_data}
#     if out_path is not None :
#         with open(out_path, 'wb') as f:
#             print('saving images to', out_path)
#             pickle.dump(data_out, f)
#     wiffFile.Close()
#     return data_out
#
# def build_image_ms2_wiff_2(path_wiff, out_path, bin_mz=1):
#     import numpy as np
#     import pickle
#     from collections import OrderedDict
#
#     wiffFile = WiffFileReader(path_wiff)
#
#     start_rt = wiffFile.GetStartTime()
#     end_rt = wiffFile.GetEndTime()
#     span_rt = end_rt - start_rt
#
#     first_scan = wiffFile.GetFirstSpectrumNumber()
#     last_scan = wiffFile.GetLastSpectrumNumber()
#
#     # ----------------------------
#     # 1. Identify MS1 range and DIA windows
#     # ----------------------------
#     dia_window_set = set()
#     max_cycle = 0
#     ms1_start_mz = None
#     ms1_end_mz = None
#
#     for scanNumber in range(first_scan, last_scan):
#         scan = wiffFile.source.ScanInfos[scanNumber]
#
#         if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
#             ms1_start_mz = scan.LowMz
#             ms1_end_mz = scan.HighMz
#             max_cycle += 1
#         else:
#             dia_window_set.add((scan.LowMz, scan.HighMz)) # MS1 is index 0
#     dia_windows = sorted(dia_window_set, key=lambda x: (x[0], x[1]))
#     window_to_index = {
#         win: i + 1  # MS1 = 0
#         for i, win in enumerate(dia_windows)
#     }
#     start_mz = ms1_start_mz
#     end_mz = ms1_end_mz
#     total_mz = end_mz - start_mz
#
#     n_bin = int(total_mz // bin_mz)
#     size_bin = total_mz / n_bin
#
#     # ----------------------------
#     # 2. Allocate images
#     # ----------------------------
#     list_img = [
#         np.zeros((max_cycle, n_bin + 1), dtype=np.float32)
#         for _ in range(len(dia_windows) + 1)
#     ]
#
#     # ----------------------------
#     # 3. Fill images
#     # ----------------------------
#     cycle = -1  # will increment on MS1
#
#     for scanNumber in range(first_scan, last_scan):
#         masses, intensities = wiffFile.GetProfileMassListFromScanNum(scanNumber)
#         scan = wiffFile.source.ScanInfos[scanNumber]
#
#         # increment cycle on MS1
#         if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
#             cycle += 1
#
#         if cycle < 0 or len(masses) == 0:
#             continue
#
#         # vectorized binning (same bins for MS1 and MS2)
#         bins = ((masses - start_mz) / size_bin).astype(int)
#         valid = (bins >= 0) & (bins <= n_bin)
#
#         line = np.zeros(n_bin + 1, dtype=np.float32)
#         np.add.at(line, bins[valid], intensities[valid])
#
#         # safe log transform
#         line = np.log10(line + 1)
#
#         if wiffFile.GetMSOrderForScanNum(scanNumber) == 1:
#             list_img[0][cycle, :] = line
#         else:
#             key = (scan.LowMz, scan.HighMz)
#             ind = window_to_index[key]
#             list_img[ind][cycle, :] = line
#
#     # ----------------------------
#     # 4. Metadata
#     # ----------------------------
#     meta_data = {
#         'n_bin': n_bin,
#         'size_bin': size_bin,
#         'start_mz': start_mz,
#         'end_mz': end_mz,
#         'max_cycle': max_cycle,
#         'dia_windows': window_to_index,
#         'start_rt': start_rt,
#         'end_rt': end_rt,
#         'span_rt': span_rt
#     }
#
#     data_out = {'image': list_img, 'metadata': meta_data}
#
#     if out_path is not None:
#         with open(out_path, 'wb') as f:
#             print('saving images to', out_path)
#             pickle.dump(data_out, f)
#
#     wiffFile.Close()
#     return data_out


def build_image_ms2_mzml(path_mzml, out_path=None, bin_mz=1.0):
    import numpy as np
    import pickle
    import pymzml

    # ----------------------------
    # 1. Load mzML
    # ----------------------------
    run = pymzml.run.Reader(path_mzml)

    # ----------------------------
    # 2. Identify MS1 range and DIA windows
    # ----------------------------
    dia_window_set = set()
    max_cycle = 0
    ms1_start_mz = None
    ms1_end_mz = None

    for spec in run:
        if spec.ms_level == 1:
            mzs = spec.mz
            if mzs is not None and len(mzs) > 0:
                if ms1_start_mz is None or min(mzs) < ms1_start_mz:
                    ms1_start_mz = min(mzs)
                if ms1_end_mz is None or max(mzs) > ms1_end_mz:
                    ms1_end_mz = max(mzs)
            max_cycle += 1
        elif spec.ms_level == 2:
            # get isolation window from mzML
            target = spec['isolation window target m/z']
            lo = spec['isolation window lower offset']
            hi = spec['isolation window upper offset']
            dia_window_set.add((target - lo, target + hi))

    dia_windows = sorted(dia_window_set, key=lambda x: (x[0], x[1]))
    window_to_index = {win: i + 1 for i, win in enumerate(dia_windows)}  # MS1 = 0

    start_mz = ms1_start_mz
    end_mz = ms1_end_mz
    total_mz = end_mz - start_mz
    n_bin = int(total_mz // bin_mz)
    size_bin = total_mz / n_bin

    # ----------------------------
    # 3. Allocate images
    # ----------------------------
    list_img = [
        np.zeros((max_cycle, n_bin + 1), dtype=np.float32)
        for _ in range(len(dia_windows) + 1)
    ]

    # ----------------------------
    # 4. Fill images
    # ----------------------------
    cycle = -1  # incremented on MS1

    for spec in run:
        mzs = spec.mz
        ints = spec.i
        if mzs is None or len(mzs) == 0:
            continue

        if spec.ms_level == 1:
            cycle += 1
        if cycle < 0:
            continue

        # binning
        masses = np.array(mzs)
        intensities = np.array(ints)
        bins = ((masses - start_mz) / size_bin).astype(int)
        valid = (bins >= 0) & (bins <= n_bin)
        line = np.zeros(n_bin + 1, dtype=np.float32)
        np.add.at(line, bins[valid], intensities[valid])
        line = np.log10(line + 1)  # safe log

        if spec.ms_level == 1:
            list_img[0][cycle, :] = line
        else:
            target = spec['isolation window target m/z']
            lo = spec['isolation window lower offset']
            hi = spec['isolation window upper offset']
            key = (target - lo, target + hi)
            ind = window_to_index[key]
            list_img[ind][cycle, :] = line

    # ----------------------------
    # 5. Metadata
    # ----------------------------
    # get first and last MS1 scans
    start_rt = None
    end_rt = None
    for spec in run:
        if spec.ms_level == 1:
            rt = spec.scan_time[0]  # get numeric RT
            if start_rt is None:
                start_rt = rt
            end_rt = rt


    meta_data = {
        'n_bin': n_bin,
        'size_bin': size_bin,
        'start_mz': start_mz,
        'end_mz': end_mz,
        'max_cycle': max_cycle,
        'dia_windows': window_to_index,
        'span_rt': end_rt - start_rt,
        'start_rt': start_rt,
        'end_rt': end_rt,
    }

    data_out = {'image': list_img, 'metadata': meta_data}

    if out_path is not None:
        with open(out_path, 'wb') as f:
            print('saving images to', out_path)
            pickle.dump(data_out, f)

    return data_out


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        f_name = f.readline()
        if not (os.path.exists(f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{f_name}.pkl')):
            try :
                data_out = build_image_ms2_mzml(f'/lustre/fsn1/projects/rech/bun/ucg81ws/mzml/{f_name}.mzML',f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{f_name}.pkl')
            except:
                print(f_name)
    # print('building image')
    # sample_list = glob.glob('/lustre/fsn1/projects/rech/bun/ucg81ws/mzml/**.mzML', recursive=True)
    # random.shuffle(sample_list)
    # sample_with_error=[]
    # for sample in sample_list:
    #     f_name = os.path.basename(sample).split('.mzML')[0]
    #     print(f_name)
    #     if not(os.path.exists(f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{f_name}.pkl')):
    #         try :
    #             data_out = build_image_ms2_mzml(f'/lustre/fsn1/projects/rech/bun/ucg81ws/mzml/{f_name}.mzML',f'/lustre/fsn1/projects/rech/bun/ucg81ws/image/{f_name}.pkl')
    #         except:
    #             sample_with_error.append(sample)
    # print('sample with error', sample_with_error)


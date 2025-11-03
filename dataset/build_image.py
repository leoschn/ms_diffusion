from pyRawMSDataReader.pyRawMSDataReader.RawFileReader import RawFileReader
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

if __name__ == '__main__':
    data_out = build_image_ms2_raw('../data/raw/20250624_ESCCOL100_VN_Microflow_100pct_ACN_15min_4Th_DIA_5ul_inj_1.raw','../data/image/ESCCOL100.pkl')
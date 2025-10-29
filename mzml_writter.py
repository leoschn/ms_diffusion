from psims.mzml.writer import MzMLWriter
import numpy as np
import pyopenms as oms

def get_scan_data():

    scans = []
    for i in range(100):
        scan = {'intensity_array':100*(10+np.random.randn(100)),'mz_array':100*(10+np.random.randn(100)),'id':i}
        products = []
        for j in range(10):
            products.append({'intensity_array':100*(10+np.random.randn(100)),'mz_array':100*(10+np.random.randn(100)),'precursor_mz':100+j,'precursor_intensity':100.,'precursor_charge':2,'precursor_scan_id':i,'id':j})
        scans.append((scan, products))
    return scans

# Load the data to write
scans = get_scan_data()

with MzMLWriter(open("out.mzML", 'wb'), close=True) as out:
    # Add default controlled vocabularies
    out.controlled_vocabularies()

    out.file_description([  # the list of file contents terms
        "MS1 spectrum",
        "MSn spectrum",
        "centroid spectrum"
    ])

    out.software_list([
        {"id": "psims-writer",
         "version": "0.1.2",
         "params": [
             "python-psims",
         ]}
    ])

    source = out.Source(1, ["electrospray ionization", "electrospray inlet"])
    analyzer = out.Analyzer(2, ["fourier transform ion cyclotron resonance mass spectrometer"])
    detector = out.Detector(3, ["inductive detector"])
    config = out.InstrumentConfiguration(id="IC1", component_list=[source, analyzer, detector],
                                            params=["LTQ-FT"])
    out.instrument_configuration_list([config])

    methods = []

    # methods.append(out.ProcessingMethod(order=1, sofware_reference="psims-writer", params=["Gaussian smoothing","median baseline reduction","MS:1000035", "Conversion to mzML"]))
    processing = out.DataProcessing(methods, id='DP1')
    out.data_processing_list([processing])

    # Open the run and spectrum list sections
    with out.run(id="my_analysis"):
        spectrum_count = len(scans) + sum([len(products) for _, products in scans])
        with out.spectrum_list(count=spectrum_count):
            for scan, products in scans:
                # Write Precursor scan
                out.write_spectrum(
                    scan['mz_array'], scan['intensity_array'],
                    scan_start_time=3.5,#RT in minutes
                    id=scan['id'], params=[
                        "MS1 Spectrum",
                        {"ms level": 1},
                        {"total ion current": sum(scan['intensity_array'])}
                     ])
                # Write MSn scans
                for prod in products:
                    out.write_spectrum(
                        prod['mz_array'], prod['intensity_array'],
                        id=prod['id'], params=[
                            "MSn Spectrum",
                            {"ms level": 2},
                            {"total ion current": sum(prod['intensity_array'])}
                         ],
                         # Include precursor information
                         precursor_information={
                            "mz": prod['precursor_mz'],
                            "intensity": prod['precursor_intensity'],
                            "charge": prod['precursor_charge'],
                            "scan_id": prod['precursor_scan_id'],
                            "activation": ["beam-type collisional dissociation", {"collision energy": 25}],
                            "isolation_window": [prod['precursor_mz'] - 1, prod['precursor_mz'], prod['precursor_mz'] + 1]
                         })

exp = oms.MSExperiment()
oms.MzMLFile().load("out.mzML", exp)
spec = exp.getSpectrum(0)
rt = spec.getRT()
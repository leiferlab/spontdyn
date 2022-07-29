import numpy as np, matplotlib.pyplot as plt
import wormdatamodel as wormdm

ds_list = "/projects/LEIFER/francesco/spontdyn_list.txt"
ds_list = wormdm.signal.file.load_ds_list(ds_list)


for folder in ds_list:
    sig = wormdm.signal.Signal.from_file(
                        folder,"gRaw",nan_interp=True,remove_spikes=False,
                        smooth=False, photobl_calc=True, photobl_appl=False)
    
    sig.to_file(folder,"gRaw")

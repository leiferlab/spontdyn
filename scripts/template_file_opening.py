import numpy as np, matplotlib.pyplot as plt
import wormdatamodel as wormdm

ds_list = "/projects/LEIFER/francesco/spontdyn_list.txt"
ds_list = wormdm.signal.file.load_ds_list(ds_list,tags="AML32",exclude_tags="505")

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":False}

sigs = []
for folder in ds_list:
    sig = wormdm.signal.Signal.from_file(folder,"gRaw",**signal_kwargs)
    sigs.append(sig)
    
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.imshow(sigs[0].data)
plt.show()

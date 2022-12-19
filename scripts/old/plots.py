import numpy as np, matplotlib.pyplot as plt, re, sys
from sklearn.decomposition import PCA
from scipy.stats import kstest
import wormdatamodel as wormdm

no_normalize = "--no-normalize" in sys.argv
normalize_recordings = "--normalize-recordings" in sys.argv and not no_normalize
normalize_neurons = not no_normalize and not normalize_recordings

ds_list_file = "/projects/LEIFER/francesco/spontdyn_list.txt"
tagss = ["488 AML32","505 AML32","488 AML70","488 AKS521.1.i","488 AKS522.1.i",]
group = [0,1,0,1,1,]
cs = ["C"+str(g) for g in group] # Color by group
cs = ["C"+str(i) for i in np.arange(len(tagss))] # Each tag its own color

signal_kwargs = {"remove_spikes": True,  "smooth": False, 
                 "nan_interp": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl": True}

# Iterate over tags
for k in np.arange(len(tagss)):
    # Get list of recordings with given tags
    tags = tagss[k]
    g = group[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)
    
    # Number of rows and columns for plots 
    n = len(ds_list)
    ncols = int(np.sqrt(n))+1
    nrows = int(np.sqrt(n))+1
    
    # Replace strain names with genotype
    tags = re.sub("AML32","wt",tags)
    tags = re.sub("AML70","lite-1",tags)
    tags = re.sub("AKS521.1.i","gur-3",tags)
    tags = re.sub("AKS522.1.i","lite-1;gur-3",tags)
    
    fig = plt.figure(k+1)
    fig.suptitle(tags)
    
    for i in np.arange(n):
        # Load files
        folder = ds_list[i]
        rec = wormdm.data.recording(folder,legacy=True,rectype="3d",settings={"zUmOverV":200./10.})
        sig = wormdm.signal.Signal.from_file(folder,"gRaw",**signal_kwargs)
        
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.imshow(sig.data.T,aspect="auto")
        
plt.show()


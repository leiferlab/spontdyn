import numpy as np, matplotlib.pyplot as plt, re, sys
from sklearn.decomposition import PCA
import wormdatamodel as wormdm

ds_list_file = "/projects/LEIFER/francesco/spontdyn_list.txt"
tagss = ["488 AML32","488 AML70","505 AML32","505 AML70","488 AKS521.1.i","488 AKS522.1.i","505+405simul AML32",]
cs = ["C0","C0","C1","C1","C2","C3","C0"]
cs = ["C"+str(i) for i in np.arange(len(tagss))]

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "nan_interp": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True}

# Define spectral ranges in which to look for oscillations
T_range = np.array([300.,30.])
f_range = 1./T_range
f = np.linspace(f_range[0],f_range[1],100)
print("spectral range",np.around(f_range,3))

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)

for k in np.arange(len(tagss)):
    tags = tagss[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)
    n = len(ds_list)
    
    # Replace strain names with genotype
    tags = re.sub("AML32","wt",tags)
    tags = re.sub("AML70","lite-1",tags)
    tags = re.sub("AKS521.1.i","gur-3",tags)
    tags = re.sub("AKS522.1.i","lite-1;gur-3",tags)
    
    avgfts = []

    for i in np.arange(n):
        folder = ds_list[i]
        rec = wormdm.data.recording(folder,legacy=True,rectype="3d",settings={"zUmOverV":200./10.})
        sig = wormdm.signal.Signal.from_file(folder,"gRaw",**signal_kwargs)
        
        ftsig = np.fft.fft(sig.data,axis=0,norm="ortho")
        ftsig = np.absolute(ftsig)**2
        f_ = np.fft.fftfreq(sig.data.shape[0],d=rec.Dt)
        
        totp = np.sum(ftsig)
        
        # Slice frequency range specified above
        f0 = np.argmin(np.abs(f-f_range[0]))
        f1 = np.argmin(np.abs(f-f_range[1]))
        
        avgft_ = np.average(ftsig,axis=1)/totp
        avgft = np.interp(f,f_[f0:f1],avgft_[f0:f1])
        avgfts.append(avgft)
        
    avgfts = np.array(avgfts)
    avgavgft = np.average(avgfts,axis=0)
    
    ax1.plot(f,avgavgft,label=tags,c=cs[k])
    
ax1.set_xlabel("frequency (Hz)")
ax1.set_ylabel("average power spectrum")
ax1.legend()
fig1.tight_layout()

plt.show()

import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wormdatamodel as wormdm

ds_list_file = "/projects/LEIFER/francesco/spontdyn_list.txt"
tagss = ["488 AML32","488 AML70","505 AML32","505 AML70","488 AKS521.1.i","505+405simul AML32"]
#cs = ["C0","C1","C0","C1","C1","C0"]
cs = ["C"+str(i) for i in np.arange(len(tagss))]

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "nan_interp": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True}

plot = False

# Define spectral ranges in which to look for oscillations
T_range = np.array([300.,30.])
f_range = 1./T_range
print("spectral range",np.around(f_range,3))


fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)

for k in np.arange(len(tagss)):
    tags = tagss[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)

    n = len(ds_list)
    ncols = int(np.sqrt(n))+1
    nrows = int(np.sqrt(n))+1

    expl_vars = []
    stdws = []
    maxfs = []
    for i in np.arange(n):
        folder = ds_list[i]
        rec = wormdm.data.recording(folder,legacy=True,rectype="3d",settings={"zUmOverV":200./10.})
        sig = wormdm.signal.Signal.from_file(folder,"gRaw",**signal_kwargs)
        
        pca = PCA()
        pcs = pca.fit_transform(sig.data)
        weights = pca.components_
        expl_var = pca.explained_variance_ratio_
        
        ftpc = np.fft.fft(pcs,axis=0,norm="ortho")
        f = np.fft.fftfreq(pcs.shape[0],d=rec.Dt)
        df = f[1]-f[0]
        
        # Integrate over frequency range specified above and make a sorter 
        f0 = np.argmin(np.abs(f-f_range[0]))
        f1 = np.argmin(np.abs(f-f_range[1]))
        p = np.sum(np.absolute(ftpc[f0:f1])**2,axis=0)*df
        sorter = np.argsort(p)[::-1]
        
        # Make figures for individual recording and store variables for global plot
        if plot:
            fig1 = plt.figure(1)
            fig2 = plt.figure(2)
            fig3 = plt.figure(3)
        for j in np.arange(9):
            jp = sorter[j]
            
            # Find peak frequency inside the spectral range
            maxf = f[f0+np.argmax(np.absolute(ftpc[f0:f1,jp]))]
            
            # Internally sort the weights of the PC to see their distribution
            weights_sorted = np.sort(np.abs(weights[jp]))[::-1]
            
            avgw = np.sum(weights_sorted*np.arange(len(weights_sorted))) / np.sum(weights_sorted)
            stdw = np.sqrt(np.sum((weights_sorted-avgw)**2)/len(weights_sorted))
            # Normalize by number of weights
            stdw /= len(weights_sorted)
            
            if j in [0]:#[0,1]: 
                stdws.append(stdw)
                expl_vars.append(expl_var[jp])
                maxfs.append(maxf)
            
            if plot:    
                ax = fig1.add_subplot(3,3,j+1)
                ax.plot(np.arange(len(pcs[:,jp]))*rec.Dt,pcs[:,jp])
                #ax.plot(f[f0:f1],np.absolute(ftpc[f0:f1,jp])**2)
                ax.set_title(str(np.around(maxf,3)))
                
                ax = fig2.add_subplot(3,3,j+1)
                ax.bar(np.arange(len(weights_sorted)),weights_sorted)
                ax.set_title(str(np.around(stdw,2)))
            
        if plot:
            ax3 = fig3.add_subplot(111)
            ax3.imshow(sig.data.T,aspect="auto")
                
            fig1.suptitle("PCs sorted by power in ("+",".join(f_range.astype(str))+") Hz")
            fig2.suptitle("Weights of PCs sorted by power in ("+",".join(f_range.astype(str))+") Hz, and internally sorted")
            fig1.tight_layout()
            fig2.tight_layout()
            plt.show()
    
    maxfs = np.array(maxfs)
    expl_vars = np.array(expl_vars)
    stdws = np.array(stdws)
    
    #maxfs = 0.5*(maxfs[::2]+maxfs[1::2])
    #expl_vars = 0.5*(expl_vars[::2]+expl_vars[1::2])
    #stdws = 0.5*(stdws[::2]+stdws[1::2])

    markersize = (stdws**3)/np.max(stdws**3)*100
    ax4.scatter(expl_vars,maxfs,s=markersize,label=tags,color=cs[k])


ax4.set_xlabel("explained variance")
ax4.set_ylabel("peak frequency (Hz)")
ax4.set_title("PC with largest power in spectral range "+str(np.around(f_range,3))+" Hz\n"+\
              "size scales with normalized standard deviation\n"+\
              "\"~what fraction of neurons are in this PC\"")
ax4.legend()
fig4.tight_layout()
plt.show()

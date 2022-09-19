import numpy as np, matplotlib.pyplot as plt, re, sys
from sklearn.decomposition import PCA
import wormdatamodel as wormdm

ds_list_file = "/projects/LEIFER/francesco/spontdyn_list.txt"
tagss = ["488 AML32","488 AML70","505 AML32","505 AML70","488 AKS521.1.i","488 AKS522.1.i","505+405simul AML32",]
cs = ["C0","C0","C1","C1","C2","C3","C0"]
tagss = ["488 AML32","488 AML70","488 AKS521.1.i","488 AKS522.1.i",]
cs = ["C0","C0","C2","C3"]
tagss = ["488 AML32","488 AML70"]#,#["505 AML32","505 AML70"]#["488 AML32","488 AML70"]#,
cs = ["C"+str(i) for i in np.arange(len(tagss))]

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "nan_interp": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True}

plot = "--plot" in sys.argv
average = "--average" in sys.argv

# Define spectral ranges in which to look for oscillations
T_range = np.array([300.,30.])
f_range = 1./T_range
print("spectral range",np.around(f_range,3))


fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)

if average:
    fig6 = plt.figure(6)
    ax6 = fig6.add_subplot(111)

for k in np.arange(len(tagss)):
    tags = tagss[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)

    n = len(ds_list)
    ncols = int(np.sqrt(n))+1
    nrows = int(np.sqrt(n))+1

    expl_vars = []
    stdws = []
    maxfs = []
    ps = []
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
            
            if (average and j in [0,1]) or (not average and j in [0]):
                stdws.append(stdw)
                expl_vars.append(expl_var[jp])
                maxfs.append(maxf)
                ps.append(p[jp])
            
            if plot:    
                ax = fig1.add_subplot(3,3,j+1)
                ax.plot(np.arange(len(pcs[:,jp]))*rec.Dt,pcs[:,jp])
                #ax.plot(f[f0:f1],np.absolute(ftpc[f0:f1,jp])**2)
                ax.set_title(str(np.around(maxf,3))+","+str(np.around(p[jp],3)))
                
                ax = fig2.add_subplot(3,3,j+1)
                ax.bar(np.arange(len(weights_sorted)),weights_sorted)
                ax.set_title(str(np.around(stdw,2)))
            
        if plot:
            ax3 = fig3.add_subplot(111)
            ax3.imshow(sig.data.T,aspect="auto")
                
            fig1.suptitle("PCs sorted by power in ("+",".join(np.around(f_range,3).astype(str))+") Hz")
            fig2.suptitle("Weights of PCs sorted by power in ("+",".join(np.around(f_range,3).astype(str))+") Hz, and internally sorted")
            fig1.tight_layout()
            fig2.tight_layout()
            plt.show()
    
    # Replace strain names with genotype
    tags = re.sub("AML32","wt",tags)
    tags = re.sub("AML70","lite-1",tags)
    tags = re.sub("AKS521.1.i","gur-3",tags)
    tags = re.sub("AKS522.1.i","lite-1;gur-3",tags)
    
    maxfs = np.array(maxfs)
    expl_vars = np.array(expl_vars)
    stdws = np.array(stdws)
    ps = np.array(ps)
    
    # Make histogram of frequencies. If average, plot both first and second PC
    # separately.
    if not average:
        ax5.hist(maxfs,label=tags,color=cs[k],alpha=0.2)
    else:
        ax5.hist(maxfs[::2],label=tags,color=cs[k],alpha=0.3)
        ax6.hist(maxfs[1::2],label=tags,color=cs[k],alpha=0.2)
    
    if average: 
        maxfs = (maxfs[::2]*ps[::2]+maxfs[1::2]*ps[1::2])/(ps[::2]+ps[1::2])
        ##maxfs = 0.5*(maxfs[::2]+maxfs[1::2])
        expl_vars = (expl_vars[::2]+expl_vars[1::2])
        stdws = (stdws[::2]+stdws[1::2])
    
    # Make scatter plot
    x = expl_vars
    y = maxfs
    m = stdws
    markersize = (m**3)/np.max(m**3)*100
    
    ax4.scatter(x,y,s=markersize,label=tags,color=cs[k])
    
    


ax4.set_xlabel("explained variance")
ax4.set_ylabel("peak frequency (Hz)")
ax4.set_title("PC with largest power in spectral range "+str(np.around(f_range,3))+" Hz\n"+\
              "size scales with normalized standard deviation\n"+\
              "\"~what fraction of neurons are in this PC\"")
ax4.legend()
fig4.tight_layout()

ax5.set_xlabel("frequency (Hz)")
ax5.set_ylabel("number")
ax5.legend()
fig5.tight_layout()

if average:
    ax6.set_xlabel("frequency (Hz)")
    ax6.set_ylabel("number")
    ax6.legend()
    fig6.tight_layout()
plt.show()

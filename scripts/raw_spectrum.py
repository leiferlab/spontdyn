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

# Define spectral ranges in which to look for oscillations
T_range = np.array([100.,30.])
f_range = 1./T_range
f = np.linspace(f_range[0],f_range[1],100)
print("spectral range",np.around(f_range,3))

# Prepare figures
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)

# Array to store peak frequencies
maxfss = np.empty(len(np.unique(group)),dtype=object)
for i in np.arange(len(maxfss)): maxfss[i] = []

# Array to store fractional power in frequency band 
fracps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(fracps)): fracps[i] = np.empty(0)

max_fracps = np.zeros(len(fracps))

# Iterate over tags
for k in np.arange(len(tagss)):
    # Get list of recordings with given tags
    tags = tagss[k]
    g = group[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)
    n = len(ds_list)
    
    # Replace strain names with genotype
    tags = re.sub("AML32","wt",tags)
    tags = re.sub("AML70","lite-1",tags)
    tags = re.sub("AKS521.1.i","gur-3",tags)
    tags = re.sub("AKS522.1.i","lite-1;gur-3",tags)
    
    avgfts = []

    for i in np.arange(n):
        # Load files
        folder = ds_list[i]
        rec = wormdm.data.recording(folder,legacy=True,rectype="3d",settings={"zUmOverV":200./10.})
        sig = wormdm.signal.Signal.from_file(folder,"gRaw",**signal_kwargs)
        
        # Compute Fourier transform
        ftsig = np.fft.fft(sig.data-np.average(sig.data,axis=0),axis=0,norm="ortho")*rec.Dt
        ftsig = np.absolute(ftsig)**2
        
        # Slice frequency range specified above
        f_ = np.fft.fftfreq(sig.data.shape[0],d=rec.Dt)
        f0 = np.argmin(np.abs(f_-f_range[0]))
        f1 = np.argmin(np.abs(f_-f_range[1]))
        df = f_[1]-f_[0]
        
        # Calculate fraction of power inside frequency range
        if normalize_neurons:    
            totp = np.sum(ftsig,axis=0)*df
        elif normalize_recordings:
            totp = np.sum(ftsig)*df
        elif no_normalize:
            totp = 1.0
            
        fracp = np.sum(ftsig[f0:f1]/totp,axis=0)*df
        #fracps[k].append(fracp)
        fracps[k] = np.append(fracps[k],fracp)
        
        # Store peak frequency 
        for m in np.arange(sig.data.shape[1]):
            maxf = f[f0+np.argmax(np.absolute(ftsig[f0:f1,m]))]
            maxfss[g].append(maxf)
        
        avgft_ = np.average(ftsig/totp,axis=1)
        avgft = np.interp(f,f_[f0:f1],avgft_[f0:f1])
        avgfts.append(avgft)
    
    max_fracps[k] = np.max(fracps[k])
        
    avgfts = np.array(avgfts)
    avgavgft = np.average(avgfts,axis=0)
    
    ax1.plot(f,np.convolve(avgavgft,np.ones(1),mode="same"),label=tags,c=cs[k])

ax1.set_xlabel("frequency (Hz)")
ax1.set_ylabel("average power spectrum")
ax1.legend()
fig1.tight_layout()

for g in np.unique(group):
    ax2.hist(maxfss[g],label="G"+str(g),color="C"+str(g),alpha=0.2,density=True)
ax2.set_xlabel("frequency (Hz)")
ax2.set_ylabel("density")
ax2.legend()
fig2.tight_layout()

# Parameters for bar plots
dn = 1.0
d = 0.2
bar_width = 0.4
dy = 0.05
lbls = []

# Indices of control datasets
aml32_488_i = tagss.index("488 AML32")
aml70_488_i = tagss.index("488 AML70")

# Starting y of the lines for significance of difference (in ax5)
max_fracps_ = np.empty(0)
for mfr in max_fracps:
    max_fracps_ = np.append(max_fracps_,mfr)
y_sgf = np.max(max_fracps_)

for k in np.arange(len(tagss)):
    tags = tagss[k]
    tags = re.sub("488","488 nm\n",tags)
    tags = re.sub("505","505 nm\n",tags)
    tags = re.sub("AML32","wt",tags)
    tags = re.sub("AML70","lite-1",tags)
    tags = re.sub("AKS521.1.i","gur-3",tags)
    tags = re.sub("AKS522.1.i","lite-1;gur-3",tags)
    
    y = fracps[k]
    
    ax3.bar(k*dn,np.average(y),color=cs[k],width=bar_width,alpha=0.6,label=tags)
    ax3.scatter(k*dn+np.random.random(len(fracps[k]))*bar_width/2 - bar_width/4, fracps[k], color=cs[k],s=0.5)
    
    #ax5.violinplot(y,positions=[k],showmeans=False,showextrema=False,quantiles=None)
    ax5.boxplot(y,positions=[k],boxprops={"color":cs[k],"linewidth":2},medianprops={"color":cs[k],"linewidth":2},widths=0.8)
    
    # Significance of difference from 488 AML32
    stars = ""
    if k!=aml32_488_i:
        _,pval = kstest(np.ravel(fracps[aml32_488_i]),np.ravel(fracps[k]),alternative="less")
        print(pval)
        if pval>0.05: stars = "n.s."
        if pval<0.05: stars = "*"
        if pval<0.01: stars = "**"
        if pval<0.001: stars = "***"
        
        if stars!="":
            # Add significance stars to bar plot.
            ax3.text(k, 0.35, stars)
        
            # In ax5, also add a line to show what is being compared.
            x1 = k
            x2 = aml32_488_i
            h = 0.02#50
            y_sgf += 2*h
            ax5.plot([x1, x1, x2, x2], [y_sgf, y_sgf+h, y_sgf+h, y_sgf], lw=1.5, color="gray")
            ax5.text(0.5*(x1+x2), y_sgf+h+h/5.0, stars, ha="center", color="gray")
    
    # Significance of difference from 488 AML70
    # Currently disabled
    stars2 = ""
    if k!=aml70_488_i and k!=aml32_488_i and False:
        _,pval2 = kstest(np.ravel(fracps[aml70_488_i]),np.ravel(fracps[k]),alternative="less")
        print(pval2)
        if pval>0.05: stars = "n.s."
        if pval<0.05: stars2 = "*"
        if pval<0.01: stars2 = "**"
        if pval<0.001: stars2 = "***"
        
        if stars2!="":
            x1 = k
            x2 = aml70_488_i
            h = 0.05#50
            
            # In ax5, also add a line to show what is being compared.
            # TODO Here the y needs to be fixed. Currently it's the same as for
            # the comparison with aml32_488, so the graphics might overlap.
            ax5.plot([x1, x1, x2, x2], [y_sgf, y_sgf+h, y_sgf+h, y_sgf], lw=1.5, c="b")
            ax5.text(0.5*(x1+x2), y_sgf+h+h/5.0, stars2, ha="center", color="b")
    
    # Plot histogram of the fractional powers
    ax4.hist(fracps[k],label=tags,alpha=0.2)
    
    # Append tags to list of xticklabels to be used below.
    lbls.append(tags)

if no_normalize:
    ax3.set_ylabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
else:
    ax3.set_ylabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
ax3.set_xticks(np.arange(len(tagss)))
ax3.set_xticklabels(lbls)
fig3.tight_layout()

if no_normalize:
    ax4.set_ylabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
else:
    ax4.set_xlabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
ax4.set_ylabel("density")
ax4.legend()
fig4.tight_layout()

ax5.set_xticks(np.arange(len(tagss)))
ax5.set_xticklabels(lbls)
#ax5.set_yticks([0,0.1,0.2,0.3])
if no_normalize:
    ax5.set_ylabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
else:
    ax5.set_ylabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
fig5.tight_layout()
fig5.savefig("/projects/LEIFER/francesco/spontdyn/power_spectrum_bars.png",dpi=300,bbox_inches="tight")

plt.show()

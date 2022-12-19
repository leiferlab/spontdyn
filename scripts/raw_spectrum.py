import numpy as np, matplotlib.pyplot as plt, re, sys, os
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import wormdatamodel as wormdm
import mistofrutta as mf


#############################################################################
#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################
#############################################################################

# Hardcoding the timestep of the recording to remove the dependency from 
# additional files that should otherwise be added to the exported_data.
Dt = 1./6.

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)

export_data = "--export-data" in sys.argv
no_normalize = "--no-normalize" in sys.argv
normalize_recordings = "--normalize-recordings" in sys.argv and not no_normalize
normalize_neurons_f0 = "--normalize-neurons-f0" in sys.argv and not no_normalize
normalize_neurons = not no_normalize and not normalize_recordings \
                    and not normalize_neurons_f0

# Folder in which to save the figures
# TODO CHANGE THIS TO THE DESIRED FOLDER
fig_dst = "/projects/LEIFER/francesco/spontdyn/"

# Folders in which to export data
exp_data_folder = "/projects/LEIFER/francesco/spontdyn/exported_data/"
exp_data_folder2 = "/projects/LEIFER/francesco/spontdyn/exported_data2/"

# File containing the list of the recordings.
# TODO CHANGE THIS TO THE LOCATION OF THIS FILE ON YOUR COMPUTER
ds_list_file = "/projects/LEIFER/francesco/spontdyn_list.txt"
tagss = ["488 AML32",
         "505 AML32",
         "488 AML70",
         "488 AKS521.1.i",
         "488 AKS522.1.i",
         "AML32H2O2 10mM"]
group = [0,1,0,1,1,0]
#cs = ["C"+str(g) for g in group] # Color by group
cs = ["C"+str(i) for i in np.arange(len(tagss))] # Each tag its own color

# TODO UNCOMMENT THE VARIABLE signal_kwargs COMMENT OUT signal_kwargs_tmac
# signal_kwargs = {"preprocess": False}
signal_kwargs_tmac = {"remove_spikes": True,  "smooth": False, 
                     "nan_interp": True, "photobl_appl": False}

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
fig5 = plt.figure(5,figsize=(12,8))
ax5 = fig5.add_subplot(111)
fig6 = plt.figure(6,figsize=(12,8))
ax6 = fig6.add_subplot(111)

# Array to store peak frequencies
maxfss = np.empty(len(np.unique(group)),dtype=object)
for i in np.arange(len(maxfss)): maxfss[i] = []

# Array to store fractional power in frequency band 
fracps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(fracps)): fracps[i] = np.empty(0)

# Array to store absolute power in frequency band 
ps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(ps)): ps[i] = np.empty(0)

# Array to store PC absolute power in frequency band 
ups = np.empty(len(tagss),dtype=object)
for i in np.arange(len(ups)): ups[i] = np.empty(0)

max_fracps = np.zeros(len(fracps))

# Iterate over tags
for k in np.arange(len(tagss)):
    # Get list of recordings with given tags
    tags = tagss[k]
    g = group[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,
                                              exclude_tags=None)
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
        
        # TODO UNCOMMENT THE FOLLOWING 2 LINES AND COMMENT THE NEXT
        #sig = wormdm.signal.Signal.from_file(folder,"activity.txt",
        #                                      **signal_kwargs)
        sig = wormdm.signal.Signal.from_file(folder,"tmac",**signal_kwargs_tmac)
        
        if export_data:
            np.savetxt(exp_data_folder+re.sub(" ","_",tags)+"-"+str(i)+".txt",
                       sig.data)
            folder_ = exp_data_folder2+folder.split("/")[-2]+"/"
            if not os.path.exists(folder_):
                os.mkdir(folder_)
            np.savetxt(folder_+"activity.txt",sig.data,
                       header='#{"method": "box", "version": "tmac",'+\
                              '"ref_index": "likely Nerve multiple"')
        
        ##############################
        # ACT ON THE INDIVIDUAL TRACES
        ##############################
        
        # Compute Fourier transform
        ftsig = np.fft.fft(sig.data-np.average(sig.data,axis=0),
                           axis=0,norm="ortho")*Dt
        ftsig = np.absolute(ftsig)**2
        
        # Slice frequency range specified above
        f_ = np.fft.fftfreq(sig.data.shape[0],d=Dt)
        f0 = np.argmin(np.abs(f_-f_range[0]))
        f1 = np.argmin(np.abs(f_-f_range[1]))
        df = f_[1]-f_[0]
        
        # Calculate the absolute power inside frequency range
        integrand = ftsig[f0:f1]
        p = np.zeros(integrand.shape[1])
        for l in np.arange(integrand.shape[1]):
            p[l] = mf.num.integral(integrand[:,l],df)
        ps[k] = np.append(ps[k],p)
        
        # Calculate fraction of power inside frequency range
        if normalize_neurons:
            totp = np.sum(ftsig,axis=0)*df
        elif normalize_neurons_f0:
            totp = np.sum(ftsig[f0:f0+3])*df
        elif normalize_recordings:
            totp = np.sum(ftsig)*df
        elif no_normalize:
            totp = 1.0
        
        fracp = p/totp
        fracps[k] = np.append(fracps[k],fracp)
        
        # Store peak frequency 
        for m in np.arange(sig.data.shape[1]):
            maxf = f[f0+np.argmax(np.absolute(ftsig[f0:f1,m]))]
            maxfss[g].append(maxf)
        
        avgft_ = np.nanmean(ftsig/totp,axis=1)
        avgft = np.interp(f,f_[f0:f1],avgft_[f0:f1])
        avgfts.append(avgft)
        
        ################
        # ACT ON THE PCS
        ################
        
        u,s,v = np.linalg.svd(sig.data-np.average(sig.data,axis=0),
                              full_matrices=False)
        u *= s # denormalize
        
        # Compute Fourier transform
        ftu = np.fft.fft(u,axis=0,norm="ortho")
        ftu = np.absolute(ftu)**2
        integrand = ftu[f0:f1]
        up = np.zeros(integrand.shape[1])
        for l in np.arange(integrand.shape[1]):
            up[l] = mf.num.integral(integrand[:,l],df)
        
        # Choose which singular vectors to use
        sel = np.arange(3) # first 3
        
        ups[k] = np.append(ups[k],up[sel])
        
    
    max_fracps[k] = np.nanmax(fracps[k])
        
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
ref_tag_i = tagss.index("488 AML32")
ttest_direction = "greater"

# Starting y of the lines for significance of difference (in ax5)
max_fracps_ = np.empty(0)
for mfr in max_fracps:
    max_fracps_ = np.append(max_fracps_,mfr)
y_sgf = np.max(max_fracps_)/4

for k in np.arange(len(tagss)):
    tags = tagss[k]
    tags = re.sub("AML32H2O2 10mM","505 WT\nH$_2$O$_2$",tags)
    tags = re.sub("488","488 nm\n",tags)
    tags = re.sub("505","505 nm\n",tags)
    tags = re.sub("AML32","WT",tags)
    tags = re.sub("AML70",r"$lite-1$",tags)
    tags = re.sub("AKS521.1.i",r"$gur-3$",tags)
    tags = re.sub("AKS522.1.i",r"$lite-1;gur-3$",tags)
    
    y = fracps[k]
    z = ups[k]
    
    ax3.bar(k*dn,np.average(y),color=cs[k],width=bar_width,alpha=0.6,label=tags)
    ax3.scatter(k*dn+np.random.random(len(fracps[k]))*bar_width/2 - bar_width/4,
                fracps[k], color=cs[k],s=0.5)
    
    parts = ax5.violinplot(y,positions=[k],showmeans=False,
                           showextrema=False,quantiles=None)
    for pc in parts['bodies']:
        pc.set_facecolor(cs[k])
        pc.set_edgecolor(cs[k])
    
    ax6.boxplot(z,positions=[k],boxprops={"color":cs[k],"linewidth":2},
                medianprops={"color":cs[k],"linewidth":2},widths=bar_width,
                showfliers=False,whis=(0,100))
    x_scatter_2 = k*dn + mf.plt.simple_beeswarm(z)*bar_width/2
    ax6.scatter(x_scatter_2,z,edgecolor=cs[k],facecolor=cs[k],s=50,alpha=0.8,)
    
    # Significance of difference from 488 AML32
    if k!=ref_tag_i:
        stats,pval = ttest_ind(np.ravel(fracps[ref_tag_i]),np.ravel(fracps[k]),
                               equal_var=False,alternative=ttest_direction)
        #print("pval", pval, "stats", stats)
        stars = mf.plt.p_to_stars(pval)
            
        y0avg = np.average(fracps[ref_tag_i])
        dyy = (np.average(fracps[k])-y0avg)/y0avg
        smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
        print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,
              "than "+tagss[ref_tag_i]+" with p",
              np.format_float_scientific(pval,1))
        
        if stars not in ["",]:
            if stars == "n.s.": stars = ""
            # Add significance stars to bar plot.
            ax3.text(k, 0.35, stars)
        
            # In ax5, also add a line to show what is being compared.
            x1 = k
            x2 = ref_tag_i
            h = 0.0002#50
            y_sgf += 2*h
            ax5.plot([x1, x1, x2, x2], [y_sgf, y_sgf+h, y_sgf+h, y_sgf], lw=1.5,
                     color="gray")
            ax5.plot(x1,y_sgf,'v',color="gray")
            signdyy = "+" if np.sign(dyy)==1 else "-"
            ax5.text(0.5*(x1+x2), y_sgf+h+h/5.0,signdyy+str(int(abs(dyy)*100))+\
                     "% "+stars, ha="center", color="gray")
            
    # Plot histogram of the fractional powers
    ax4.hist(fracps[k],label=tags,alpha=0.2)
    
    # Append tags to list of xticklabels to be used below.
    lbls.append(tags)
    
# Add relative increase due to H2O2, just as above for the general control
k = tagss.index("AML32H2O2 10mM")
k2 = tagss.index("505 AML32")
ttest_direction = "less"
stats,pval = ttest_ind(np.ravel(fracps[k2]),np.ravel(fracps[k]),equal_var=False,
                       alternative=ttest_direction)
#print("pval", pval, "stats", stats)
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(fracps[k2])
dyy = (np.average(fracps[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[ref_tag_i]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    h = 0.0002
    y_sgf += 2*h
    ax5.plot([x1, x1, x2, x2], [y_sgf, y_sgf+h, y_sgf+h, y_sgf], lw=1.5,
             color="gray")
    ax5.plot(x1,y_sgf,'v',color="gray")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax5.text(0.5*(x1+x2), y_sgf+h+h/5.0, signdyy+str(int(abs(dyy)*100))+"% "+\
             stars, ha="center", color="gray")


if no_normalize:
    ax3.set_ylabel("power in "+str(np.around(f_range,3))+" Hz")
else:
    ax3.set_ylabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
ax3.set_xticks(np.arange(len(tagss)))
ax3.set_xticklabels(lbls)
fig3.tight_layout()

if no_normalize:
    ax4.set_ylabel("power in "+str(np.around(f_range,3))+" Hz")
else:
    ax4.set_xlabel("fraction of power in "+str(np.around(f_range,3))+" Hz")
ax4.set_ylabel("density")
ax4.legend()
fig4.tight_layout()

ax5.set_xticks(np.arange(len(tagss)))
ax5.set_xticklabels(lbls)
ax5.spines.right.set_visible(False)
ax5.spines.top.set_visible(False)
if no_normalize:
    ax5.set_ylabel("Power in "+str(np.around(f_range,3))+" Hz")
else:
    ax5.set_ylabel("Fraction of power in "+str(np.around(f_range,3))+" Hz")
fig5.tight_layout()
fig5.savefig(fig_dst+"power_spectrum_bars.pdf",dpi=300,bbox_inches="tight")
fig5.savefig(fig_dst+"power_spectrum_bars.png",dpi=300,bbox_inches="tight")

ax6.set_xticks(np.arange(len(tagss)))
ax6.set_xticklabels(lbls)
ax6.set_yticks([0,0.5,1,])
ax6.set_yticklabels(["0","0.5","1",],fontsize=18)
ax6.spines.right.set_visible(False)
ax6.spines.top.set_visible(False)
if no_normalize:
    ax6.set_ylabel("Power in "+str(np.around(f_range,3))+\
                   " Hz\nin each of the top 3 PCs",)
fig6.tight_layout()
fig6.savefig(fig_dst+"power_spectrum_bars_PCs.pdf",dpi=300,bbox_inches="tight")
fig6.savefig(fig_dst+"power_spectrum_bars_PCs.png",dpi=300,bbox_inches="tight")

plt.show()

import numpy as np, matplotlib.pyplot as plt, re, sys, os
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, kstest
from scipy.signal import savgol_filter as savgol
import wormdatamodel as wormdm
import mistofrutta as mf
import jPCA
from scipy.stats import mannwhitneyu
#import statsmodels.stats.multitest as smt
import mne.stats as ms




#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
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
fig_dst = "/projects/LEIFER/Wayan/LightEvokedDynamics/figures manuscript/"


# Folders in which to export data
# TODO CHANGE THIS TO THE DESIRED FOLDER
exp_data_folder = "/projects/LEIFER/Wayan/LightEvokedDynamics/figures manuscript/exported_data/"
exp_data_folder2 = "/projects/LEIFER/Wayan/LightEvokedDynamics/figures manuscript/exported_data2/"



# list of median recordings for plotting
# TODO CHANGE THIS TO THE LOCATION OF THIS FILE ON YOUR COMPUTER
# Likely something like <>/exported_data2/BrainScanner<>
tempPC_folders = ["/projects/LEIFER/PanNeuronal/20230326/BrainScanner20230326_145406/",
           "/projects/LEIFER/PanNeuronal/20230314/BrainScanner20230314_160321/",
           "/projects/LEIFER/PanNeuronal/20230404/BrainScanner20230404_182215/",
           "/projects/LEIFER/PanNeuronal/20221108/BrainScanner20221108_184729/"
           ]

# File containing the list of the recordings.
# TODO CHANGE THIS TO THE LOCATION OF THIS FILE ON YOUR COMPUTER
ds_list_file = "/projects/LEIFER/Wayan/Code/spontdyn/spontdyn_list_sort.txt"
tagss = ["488 AML32",
         "505 AML32",
         "488 AML70",
         "488 AKS521.1.i",
         "488 AKS522.1.i",
         "AML32H2O2 10mM",
	 "AKS521.1.iH2O2 10mM"]


#group = [0,1,0,1,1,0]
group = np.arange(len(tagss))
#cs = ["C"+str(g) for g in group] # Color by group
cs = ["C"+str(i) for i in np.arange(len(tagss))] # Each tag its own color

# Convert tags to plot-friendly labels
lbls = []
for k in np.arange(len(tagss)):
    tags = tagss[k]
    tags = re.sub("AML32H2O2 10mM","505 WT\nH$_2$O$_2$",tags)
    tags = re.sub("AKS521.1.iH2O2 10mM","505 $gur-3$\nH$_2$O$_2$",tags)
    tags = re.sub("488","488 nm\n",tags)
    tags = re.sub("505","505 nm\n",tags)
    tags = re.sub("AML32","WT",tags)
    tags = re.sub("AML70",r"$lite-1$",tags)
    tags = re.sub("AKS521.1.i",r"$gur-3$",tags)
    tags = re.sub("AKS522.1.i",r"$lite-1;gur-3$",tags)
    
    lbls.append(tags)

# TODO UNCOMMENT THE VARIABLE signal_kwargs COMMENT OUT signal_kwargs_tmac
# signal_kwargs = {"preprocess": False}
signal_kwargs_tmac = {"remove_spikes": True,  "smooth": False, 
                     "nan_interp": True, "photobl_appl": False}

# Define spectral ranges in which to look for oscillations
T_range = np.array([150,30])
f_range = 1./T_range
f = np.linspace(f_range[0],f_range[1],100)
print("spectral range",np.around(f_range,3))
# Define spctral range for plotting the spectral power in Fig S2A
f_plot = np.linspace(0.0002,.045,120)


# Prepare figures
fig1 = plt.figure(1,figsize=(12,8))
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(2,figsize=(12,8))
ax2 = fig2.add_subplot(111)
fig3 = plt.figure(3,figsize=(12,8))
ax3 = fig3.add_subplot(111)
fig4 = plt.figure(4,figsize=(12,8))
ax4 = fig4.add_subplot(111)
fig5 = plt.figure(5,figsize=(12,8))
ax5 = fig5.add_subplot(111)



# Array to store fractional power in frequency band 
fracps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(fracps)): fracps[i] = np.empty(0)

# Array to store average fractional power in frequency band per worm
avgfracp = np.empty(len(tagss),dtype=object)
for i in np.arange(len(avgfracp)): avgfracp[i] = np.empty(0)

max_avgfracp = np.zeros(len(avgfracp))

# Array to store absolute power in frequency band 
ps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(ps)): ps[i] = np.empty(0)

# Array to store PC absolute power in frequency band 
# up: u=singular vector, p=spectral Power
ups = np.empty(len(tagss),dtype=object)
for i in np.arange(len(ups)): ups[i] = np.empty(0)

max_fracps = np.zeros(len(fracps))
max_ups = np.zeros(len(ups))

# Array to store the proportion of neurons per worm that have a fraction of power
# inside the frequency range larger that a given threshold
threshold = 0.2
frac_neurons = np.empty(len(tagss),dtype=object)
for i in np.arange(len(frac_neurons)): frac_neurons[i] = np.empty(0)

max_frac_neuron = np.zeros(len(frac_neurons))

# Array to store the frequency below which resides a given % of the total spectral power
freq_below = np.empty(len(tagss),dtype=object)
for i in np.arange(len(freq_below)): freq_below[i] = np.empty(0)

max_freq_below = np.zeros(len(freq_below))
spectral_edge = 0.5

# Array to store median metric per genotype
median_metric = np.array([])
median_1F = np.array([])


################################################################################
# Compute metrics
################################################################################

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
        f0 = np.argmin(np.abs(f_- f_range[0]))
        f1 = np.argmin(np.abs(f_- f_range[1]))
        df = f_[1]-f_[0]

        # Index of the min max in the shortest recording
        fmin = 0
        fmax = 1450
        
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
        fracp_no_nan = fracp[np.logical_not(np.isnan(fracp))]
        fracps[k] = np.append(fracps[k],fracp_no_nan)
        avgfracp[k] = np.append(avgfracp[k],np.nanmean(fracp))
    
       
	# Calculate proportion of neurons per worm that have a fraction of power inside frequency range
	# higher than a given threshold
        frac_neuron = float(np.sum(fracp>threshold))/fracp.shape[0]
        frac_neurons[k] = np.append(frac_neurons[k], frac_neuron*100)

        # Calculate the frequency below which resides 50% of the total spectral power 
        avgft_ = np.nanmean(ftsig,axis=1)
        avgft_ = avgft_/np.sum(avgft_)
        avgft = np.interp(f_plot,f_[fmin:fmax],avgft_[fmin:fmax])
        cdf= np.cumsum(avgft)
        cdf = cdf/np.sum(avgft)
        freq_below[k] = np.append(freq_below[k],f_plot[np.where(cdf<=spectral_edge)][-1])
        avgfts.append(avgft)

        # extract median point for each genotype and corresponding value in FIG 1F
        if folder in tempPC_folders:
        	median_metric = np.append(median_metric,np.nanmean(fracp))
        	median_1F = np.append(median_1F, f_plot[np.where(cdf<=spectral_edge)][-1])

        
        
        ##############################
        # ACT ON THE PCS
        ##############################
        use_jpca = False
        if use_jpca:
            num_comp = 1
            jpca = jPCA.JPCA(num_jpcs=num_comp)
            #reshaped_data = [np.array([a]).T for a in sig.data.T]
            u, _, _, jvc = jpca.fit([sig.data,],
                                  pca=False,
                                  times=(np.arange(sig.data.shape[0])*Dt).tolist(),
                                  tstart=0,
                                  tend=(sig.data.shape[0]-1)*Dt,
                                  align_axes_to_data=False
                                  )
            u = np.array(u[0])*jvc

        else:
            # Compute regular SVD
            num_comp = 2
            data_ = sig.data-np.average(sig.data,axis=0)
            u,s,v = np.linalg.svd(data_,full_matrices=False)
            u *= s # denormalize
            
            # Compute SVD of Fourier transform
            data2_ = np.power(np.absolute(np.fft.fft(data_,axis=0)),2)
            freq2 = np.fft.fftfreq(data_.shape[0],d=Dt)
            u2,s2,v2 = np.linalg.svd(data2_,full_matrices=False)
            u2 *= s2
        
        # Compute power inside range for the PC
        # Select from what quantity
        from_ft_of_u = False
        from_u_of_ft = True
        if from_ft_of_u:
            # Compute Fourier transform of the regular SVD
            ftu = np.fft.fft(u,axis=0,norm="ortho")
            ftu = np.absolute(ftu)**2
            integrand = ftu[f0:f1]
            up = np.zeros(integrand.shape[1])
            for l in np.arange(integrand.shape[1]):
                up[l] = mf.num.integral(integrand[:,l],df)
        elif from_u_of_ft:
            # Use the SVD of the Fourier transform of the data
            integrand = np.absolute(u2[f0:f1])
            up = np.zeros(integrand.shape[1])
            for l in np.arange(integrand.shape[1]):
                up[l] = mf.num.integral(integrand[:,l],df)
                
        # Choose which singular vectors to use
        sel = np.argsort(up[:3])[::-1][:num_comp] # Takes the first num_comp PCs with the highest power
        #sel = np.arange(num_comp) # Takes the first num_comp PCs
        #try:
        #    printed
        #except:
        #    print("Using the "+str(num_comp)+" PCs with the largest power.")
        #    printed=True

        ups[k] = np.append(ups[k],(up[sel[0]]+up[sel[1]])/2)  # mean of the first 2 PCs with the most power in the frequency range 
        #ups[k] = np.append(ups[k],up[sel])    # first num_comp PC    
         
        
    max_fracps[k] = np.nanmax(fracps[k])
    max_ups[k] = np.nanmax(ups[k])
    max_frac_neuron[k] = np.nanmax(frac_neurons[k]) 
    max_avgfracp[k] = np.nanmax(avgfracp[k])
    max_freq_below[k] = np.nanmax(freq_below[k]) 
        
    avgfts = np.array(avgfts)
    avgavgft = np.average(avgfts,axis=0)
    cdf= np.cumsum(avgavgft)
    cdf = cdf/np.sum(avgavgft)    
    
    ax2.plot(f_plot,avgavgft,label=tags,c=cs[k], lw = 2) # normalized power vs frequency
    ax3.plot(f_plot,cdf,label=tags,c=cs[k],lw = 2) # CDF vs frequency    
    

# Parameters for box plots
dn = 1.0
d = 0.2
bar_width = 0.4
dy = 0.05   

# Indices of control datasets
ref_tag_i = tagss.index("488 AML32")

#Initialize plotting of significance stars and arrows
# FIG 1: Average fraction of power in frequency band
max_avgfracp_ = np.empty(0)
for mfr in max_avgfracp:
    max_avgfracp_ = np.append(max_avgfracp_,mfr)
y_sgf_avgfracp = np.max(max_avgfracp_)
h_avgfracp = y_sgf_avgfracp*0.025

#FIG 4:  frequency below which resides 50% of the total spectral power
max_freq_below_ = np.empty(0)
for mfr in max_freq_below:
    max_freq_below_ = np.append(max_freq_below_,mfr)
y_sgf_freq = np.max(max_freq_below_)
h_freq = y_sgf_freq*0.025


# FIG 5: Fraction of neuron above threshold in frequency range
max_fracn_ = np.empty(0)
for mfr in max_frac_neuron:
    max_fracn_ = np.append(max_fracn_,mfr)
y_sgf_fn = np.max(max_fracn_)
h_fn = y_sgf_fn*0.025





################################################################################
# COMPUTE P-VALUES for significance of difference from 488 AML32
################################################################################
exclude_H2O2 = tagss.index("AML32H2O2 10mM")
exclude_H2O2_gur3 = tagss.index("AKS521.1.iH2O2 10mM")
test_direction = "greater"

pval_1E = []
pval_1F = []
pval_S2B = []
for k in np.arange(len(tagss)):
    if (k!=ref_tag_i) and (k!= exclude_H2O2_gur3) and (k!= exclude_H2O2):
        _,pval = mannwhitneyu(np.ravel(avgfracp[ref_tag_i]),np.ravel(avgfracp[k]), method="exact",alternative = test_direction)
        pval_1E.append(pval)

        _,pval = mannwhitneyu(np.ravel(freq_below[ref_tag_i]),np.ravel(freq_below[k]), method="exact",alternative = test_direction)
        pval_1F.append(pval)

        _,pval = mannwhitneyu(np.ravel(frac_neurons[ref_tag_i]),np.ravel(frac_neurons[k]), method="exact",alternative = test_direction)
        pval_S2B.append(pval)




################################################################################
# COMPUTE P-VALUES ALSO FOR THE RELATIVE INCREASE OF 505 WT -> 505 WT + H202
################################################################################
k = tagss.index("AML32H2O2 10mM")
k2 = tagss.index("505 AML32")
test_direction = "less"

_,pval = mannwhitneyu(np.ravel(avgfracp[k2]),np.ravel(avgfracp[k]), method="exact",alternative = test_direction)
pval_1E.append(pval)

_,pval = mannwhitneyu(np.ravel(freq_below[k2]),np.ravel(freq_below[k]), method="exact",alternative = test_direction)
pval_1F.append(pval)

_,pval = mannwhitneyu(np.ravel(frac_neurons[k2]),np.ravel(frac_neurons[k]), method="exact",alternative = test_direction)
pval_S2B.append(pval)



################################################################################
# COMPUTE P-VALUES ALSO TO COMPARE 505 WT + H202 -> 505 gur-3 + H202
################################################################################
k = tagss.index("AKS521.1.iH2O2 10mM")
k2 = tagss.index("AML32H2O2 10mM")
test_direction = "greater"

_,pval = mannwhitneyu(np.ravel(avgfracp[k2]),np.ravel(avgfracp[k]), method="exact",alternative = test_direction)
pval_1E.append(pval)

_,pval = mannwhitneyu(np.ravel(freq_below[k2]),np.ravel(freq_below[k]), method="exact",alternative = test_direction)
pval_1F.append(pval)

_,pval = mannwhitneyu(np.ravel(frac_neurons[k2]),np.ravel(frac_neurons[k]), method="exact",alternative = test_direction)
pval_S2B.append(pval)



################################################################################
# CORRECT THE P-VALUES FOR  MULTIPLE HYPOTHESES VIA THE BENJAMINI HOCHBERG PROCEDURE
################################################################################

_,pval_corr_1E = ms.fdr_correction(pval_1E, alpha=0.05, method='indep')
_,pval_corr_1F = ms.fdr_correction(pval_1F, alpha=0.05, method='indep')
_,pval_corr_S2B = ms.fdr_correction(pval_S2B, alpha=0.05, method='indep')


################################################################################
# PLOTTING
################################################################################
median_index = np.array([5,6,6,4])
F1E = []
F1F = []
for k in np.arange(len(tagss)):
    tags = tagss[k]
    tags = re.sub("AML32H2O2 10mM","505 WT\nH$_2$O$_2$",tags)
    tags = re.sub("488","488 nm\n",tags)
    tags = re.sub("505","505 nm\n",tags)
    tags = re.sub("AML32","WT",tags)
    tags = re.sub("AML70",r"$lite-1$",tags)
    tags = re.sub("AKS521.1.i",r"$gur-3$",tags)
    tags = re.sub("AKS522.1.i",r"$lite-1;gur-3$",tags)
    #tags = re.sub("AKS521.1.iH2O2 10mM","505 AKS521.1.i\nH$_2$O$_2$",tags)

    # FIG 1 Average fraction of power in frequency band
    ax1.boxplot(avgfracp[k],positions=[k],boxprops={"color":cs[k],"linewidth":2}, 
                meanline = True,meanprops={"color":cs[k],"linewidth":2,"linestyle":'solid'},
                medianprops={"linewidth":0},showmeans = True, widths=bar_width,
                showfliers=False, whiskerprops = {"linewidth":0},capprops= {"linewidth":0}) 
    means = np.mean(avgfracp[k])
    std = np.std(avgfracp[k])

    ax1.errorbar(k,means, std, ecolor = 'black', capsize = 6, fmt = '.k')
    x_scatter_avgfracp = k*dn + mf.plt.simple_beeswarm(avgfracp[k])*bar_width/2
    if (k!=2) and (k!=4) and (k!=6):
    	F1E.append(x_scatter_avgfracp)
    ax1.scatter(x_scatter_avgfracp,avgfracp[k],edgecolor=cs[k],facecolor=cs[k],s=50,alpha=0.8,)

    # Significance of difference from 488 AML32
    if (k!=ref_tag_i) and (k!= exclude_H2O2_gur3) and (k!= exclude_H2O2):
        pval = pval_corr_1E[k-1]
        stars = mf.plt.p_to_stars(pval)
        y0avg = np.average(avgfracp[ref_tag_i])
        dyy = (np.average(avgfracp[k])-y0avg)/y0avg
        smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
        print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,
              "than "+tagss[ref_tag_i]+" with p",
              np.format_float_scientific(pval,1))
        
        if stars not in ["",]:
            if stars == "n.s.": stars = "p="+str(np.around(pval,2))
            #  add a line to show what is being compared.
            x1 = k
            x2 = ref_tag_i
            y_sgf_avgfracp += 2*h_avgfracp
            ax1.plot([x1, x1, x2, x2], [y_sgf_avgfracp, y_sgf_avgfracp+h_avgfracp, y_sgf_avgfracp+h_avgfracp, y_sgf_avgfracp], 
                        lw=1.5, color="black")
            ax1.plot(x1,y_sgf_avgfracp,'v',color="black")
            signdyy = "+" if np.sign(dyy)==1 else "-"
            ax1.text(0.5*(x1+x2), y_sgf_avgfracp+h_avgfracp+h_avgfracp/5.0,
                        signdyy+str(int(abs(dyy)*100))+"% "+stars,
                        ha="center", color="black", fontsize = 12)

    
    # Figure 4 frequency below which resides 50% of the spectral power
    freq = freq_below[k]
    ax4.boxplot(freq,positions=[k],boxprops={"color":cs[k],"linewidth":2}, 
                meanline = True,meanprops={"color":cs[k],"linewidth":2,"linestyle":'solid'},
                medianprops={"linewidth":0},showmeans = True, widths=bar_width,
                showfliers=False, whiskerprops = {"linewidth":0},capprops= {"linewidth":0}) 
    means = np.mean(freq_below[k])
    std = np.std(freq_below[k])

    ax4.errorbar(k,means, std, ecolor = 'black', capsize = 6, fmt = '.k')
    x_scatter_freq = k*dn + mf.plt.simple_beeswarm(freq)*bar_width/2
    if (k!=2) and (k!=4) and (k!=6):
    	F1F.append(x_scatter_freq)
    ax4.scatter(x_scatter_freq,freq,edgecolor=cs[k],facecolor=cs[k],s=50,alpha=0.8,) 

    # Significance of difference from 488 AML32
    if (k!=ref_tag_i) and (k!= exclude_H2O2_gur3) and (k!= exclude_H2O2):
        pval = pval_corr_1F[k-1]
        stars = mf.plt.p_to_stars(pval)
        y0avg = np.average(freq_below[ref_tag_i])
        dyy = (np.average(freq_below[k])-y0avg)/y0avg
        smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
        print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,
              "than "+tagss[ref_tag_i]+" with p",
              np.format_float_scientific(pval,1))
        
        if stars not in ["",]:
            if stars == "n.s.": stars = "p="+str(np.around(pval,2))
            # add a line to show what is being compared.
            x1 = k
            x2 = ref_tag_i
            y_sgf_freq += 2*h_freq
            ax4.plot([x1, x1, x2, x2], [y_sgf_freq, y_sgf_freq+h_freq, y_sgf_freq+h_freq, y_sgf_freq], 
                        lw=1.5, color="black")
            ax4.plot(x1,y_sgf_freq,'v',color="black")
            signdyy = "+" if np.sign(dyy)==1 else "-"
            ax4.text(0.5*(x1+x2), y_sgf_freq+h_freq+h_freq/5.0,
                        signdyy+str(int(abs(dyy)*100))+"% "+stars,
                        ha="center", color="black", fontsize=14)

    # FIG5 Proportion of neurons with fraction of power above threshold
    fn = frac_neurons[k]
    ax5.boxplot(fn,positions=[k],boxprops={"color":cs[k],"linewidth":2}, 
                meanline = True,meanprops={"color":cs[k],"linewidth":2,"linestyle":'solid'},
                medianprops={"linewidth":0},showmeans = True, widths=bar_width,
                showfliers=False, whiskerprops = {"linewidth":0},capprops= {"linewidth":0}) 
    means = np.mean(frac_neurons[k])
    std = np.std(frac_neurons[k])

    ax5.errorbar(k,means, std, ecolor = 'black', capsize = 6, fmt = '.k')
    x_scatter_fn = k*dn + mf.plt.simple_beeswarm(fn)*bar_width/2
    ax5.scatter(x_scatter_fn,fn,edgecolor=cs[k],facecolor=cs[k],s=50,alpha=0.8,)

    # Significance of difference from 488 AML32
    if (k!=ref_tag_i) and (k!= exclude_H2O2_gur3) and (k!= exclude_H2O2):
        pval = pval_corr_S2B[k-1]
        stars = mf.plt.p_to_stars(pval)
        y0avg = np.average(frac_neurons[ref_tag_i])
        dyy = (np.average(frac_neurons[k])-y0avg)/y0avg
        smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
        print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,
              "than "+tagss[ref_tag_i]+" with p",
              np.format_float_scientific(pval,1))
        
        if stars not in ["",]:
            if stars == "n.s.": stars = "p="+str(np.around(pval,2))
            # add a line to show what is being compared.
            x1 = k
            x2 = ref_tag_i
            y_sgf_fn += 2*h_fn
            ax5.plot([x1, x1, x2, x2], [y_sgf_fn, y_sgf_fn+h_fn, y_sgf_fn+h_fn, y_sgf_fn], 
                        lw=1.5, color="black")
            ax5.plot(x1,y_sgf_fn,'v',color="black")
            signdyy = "+" if np.sign(dyy)==1 else "-"
            ax5.text(0.5*(x1+x2), y_sgf_fn+h_fn+h_fn/5.0,
                        signdyy+str(int(abs(dyy)*100))+"% "+stars,
                        ha="center", color="black", fontsize = 12)


    
################################################################################
# PLOT P-VALUES FOR THE RELATIVE INCREASE OF 505 WT -> 505 WT + H202
################################################################################
k = tagss.index("AML32H2O2 10mM")
k2 = tagss.index("505 AML32")

# FIG1 
pval = pval_corr_1E[4]
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(avgfracp[k2])
dyy = (np.average(avgfracp[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_avgfracp += 2*h_avgfracp
    ax1.plot([x1, x1, x2, x2], [y_sgf_avgfracp, y_sgf_avgfracp+h_avgfracp, y_sgf_avgfracp+h_avgfracp, y_sgf_avgfracp], lw=1.5,
             color="black")
    ax1.plot(x1,y_sgf_avgfracp,'v',color="black")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax1.text(0.5*(x1+x2), y_sgf_avgfracp+h_avgfracp+h_avgfracp/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="black", fontsize = 12)


# Fig 4 
pval = pval_corr_1F[4]
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(freq_below[k2])
dyy = (np.average(freq_below[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_freq += 2*h_freq
    ax4.plot([x1, x1, x2, x2], [y_sgf_freq, y_sgf_freq+h_freq, y_sgf_freq+h_freq, y_sgf_freq], lw=1.5,
             color="black")
    ax4.plot(x1,y_sgf_freq,'v',color="black")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax4.text(0.5*(x1+x2), y_sgf_freq+h_freq+h_freq/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="black", fontsize=14)


# FIG 5 
pval = pval_corr_S2B[4]
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(frac_neurons[k2])
dyy = (np.average(frac_neurons[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_fn += 2*h_fn
    ax5.plot([x1, x1, x2, x2], [y_sgf_fn, y_sgf_fn+h_fn, y_sgf_fn+h_fn, y_sgf_fn], lw=1.5,
             color="black")
    ax5.plot(x1,y_sgf_fn,'v',color="black")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax5.text(0.5*(x1+x2), y_sgf_fn+h_fn+h_fn/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="black", fontsize = 12)


################################################################################
# PLOT P-VALUES ALSO TO COMPARE 505 WT + H202 -> 505 gur-3 + H202
################################################################################
k = tagss.index("AKS521.1.iH2O2 10mM")
k2 = tagss.index("AML32H2O2 10mM")

## FIG 1
pval = pval_corr_1E[-1]
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(avgfracp[k2])
dyy = (np.average(avgfracp[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_avgfracp += 2*h_avgfracp
    ax1.plot([x1, x1, x2, x2], [y_sgf_avgfracp, y_sgf_avgfracp+h_avgfracp, y_sgf_avgfracp+h_avgfracp, y_sgf_avgfracp], lw=1.5,
             color="black")
    ax1.plot(x1,y_sgf_avgfracp,'v',color="black")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax1.text(0.5*(x1+x2), y_sgf_avgfracp+h_avgfracp+h_avgfracp/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="black", fontsize = 12)

# Fig 4
pval = pval_corr_1F[-1]
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(freq_below[k2])
dyy = (np.average(freq_below[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_freq += 2*h_freq
    ax4.plot([x1, x1, x2, x2], [y_sgf_freq, y_sgf_freq+h_freq, y_sgf_freq+h_freq, y_sgf_freq], lw=1.5,
             color="black")
    ax4.plot(x1,y_sgf_freq,'v',color="black")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax4.text(0.5*(x1+x2), y_sgf_freq+h_freq+h_freq/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="black", fontsize=14)

## FIG 5 
pval = pval_corr_S2B[-1]
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(frac_neurons[k2])
dyy = (np.average(frac_neurons[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_fn += 2*h_fn
    ax5.plot([x1, x1, x2, x2], [y_sgf_fn, y_sgf_fn+h_fn, y_sgf_fn+h_fn, y_sgf_fn], lw=1.5,
             color="black")
    ax5.plot(x1,y_sgf_fn,'v',color="black")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax5.text(0.5*(x1+x2), y_sgf_fn+h_fn+h_fn/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="black", fontsize = 12)



################################################
# FINALIZE PLOTS
################################################

# Plot the median recording based on the average fraction of power in frequency band
# as a black square in Fig 1E and F
for k in range(len(median_metric)):
	f1E = F1E[k]
	f1F = F1F[k]
	ax1.scatter(f1E[median_index[k]],median_metric[k], marker = "s",edgecolor='black',facecolor='black',s=50,alpha=1,)
	ax4.scatter(f1F[median_index[k]],median_1F[k], marker = "s",edgecolor='black',facecolor='black',s=50,alpha=1,)


ax1.set_xticks(np.arange(len(tagss)))
ax1.set_xticklabels([])
ax1.set_xticklabels(lbls)
ax1.spines.right.set_visible(False)
ax1.spines.top.set_visible(False)
ax1.set_ylabel("Average fraction of power in "+str(np.around(f_range,3))+" Hz")
fig1.savefig(fig_dst+"Average fraction of power in "+str(np.around(f_range,3))+" Hz.pdf",dpi=1200,bbox_inches="tight")
fig1.savefig(fig_dst+"Average fraction of power in "+str(np.around(f_range,3))+" Hz.png",dpi=1200,bbox_inches="tight")

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Normalized PSD")
ax2.legend(bbox_to_anchor = (0.72236,1), prop = {'size':19 })
ax2.spines.right.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.set_ylim(0,0.2)
ax2.set_yticks([0,0.1,0.21,])
#ax2.set_xticklabels([])
#ax2.set_yticklabels([])
ax2.set_xlim(0,0.045)
ax2.axvline(x=0.0066, color = 'black',ls = '--')
ax2.axvline(x=0.033, color = 'black',ls = '--')
fig2.tight_layout()
fig2.savefig(fig_dst+"Power_spectrum.pdf",dpi=300,bbox_inches="tight")
fig2.savefig(fig_dst+"Pow er_spectrum.png",dpi=300,bbox_inches="tight")



ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("CDF")
ax3.spines.right.set_visible(False)
ax3.spines.top.set_visible(False)
ax3.set_ylim(0,1.01)
ax3.set_yticks([0,0.5,1.0,])
#ax3.set_xticklabels([])
#ax3.set_yticklabels([])
ax3.set_xlim(0,0.045)
ax3.axvline(x=0.0066, color = 'black',ls = '--')
ax3.axvline(x=0.033, color = 'black',ls = '--')
fig3.tight_layout()
fig3.savefig(fig_dst+"Cumulative_power.pdf",dpi=300,bbox_inches="tight")
fig3.savefig(fig_dst+"Cumulative_power.png",dpi=300,bbox_inches="tight")


ax4.set_ylabel("Frequency below which resides "+ str(np.round(spectral_edge*100))+"% of the spectral power")
ax4.set_xticks(np.arange(len(tagss)))
ax4.set_xticklabels(lbls)
#ax4.set_xticklabels([])
#ax4.set_yticklabels([])
ax4.spines.right.set_visible(False)
ax4.spines.top.set_visible(False)
fig4.tight_layout()
fig4.savefig(fig_dst+"Spectral_edge.pdf",dpi=300,bbox_inches="tight")
fig4.savefig(fig_dst+"Spectral_edge.png",dpi=300,bbox_inches="tight")


ax5.set_xticks(np.arange(len(tagss)))
ax5.set_xticklabels(lbls)
#ax5.set_xticklabels([])
#ax5.set_yticklabels([])
ax5.spines.right.set_visible(False)
ax5.spines.top.set_visible(False)
ax5.set_ylabel("% of neurons with fraction of power in "+str(np.around(f_range,3))+\
               " Hz\n> "+str(threshold),)

fig5.tight_layout()
fig5.savefig(fig_dst+"fraction_neurons_oscillating_power_"+str(np.around(f_range,3))+".pdf",dpi=1200,bbox_inches="tight")
fig5.savefig(fig_dst+"fraction_neurons_oscillating_power_"+str(np.around(f_range,3))+".png",dpi=1200,bbox_inches="tight")


plt.show()









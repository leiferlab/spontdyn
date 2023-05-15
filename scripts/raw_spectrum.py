import numpy as np, matplotlib.pyplot as plt, re, sys, os
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, kstest
from scipy.signal import savgol_filter as savgol
import wormdatamodel as wormdm
import mistofrutta as mf
import jPCA
import utilities


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
normalize_range2 = "--normalize-range2" in sys.argv
normalize_recordings = "--normalize-recordings" in sys.argv and not no_normalize and not normalize_range2
normalize_neurons_f0 = "--normalize-neurons-f0" in sys.argv and not no_normalize and not normalize_range2
normalize_neurons = not no_normalize and not normalize_recordings \
                    and not normalize_neurons_f0
use_jpca = "--use-jpca" in sys.argv      
if use_jpca:
    print("*Using jPCA (should be num_comp = 2")
# Select from what quantity to compute the power
from_u_of_ft = "--power-from-u-of-ft" in sys.argv
from_ft_of_u = not from_u_of_ft

num_comp = 1
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--num-comp": num_comp=int(sa[1])
    
# Define spectral ranges in which to look for oscillations
T_range = np.array([100.,30.])
f_range = 1./T_range
f = np.linspace(f_range[0],f_range[1],100)
print("*spectral range",np.around(f_range,3))
T_range2 = np.array([500.,100.])
f_range2 = 1./T_range2
f2 = np.linspace(f_range2[0],f_range2[1],100)
print("*spectral range2",np.around(f_range2,3))


if from_u_of_ft: print("*Using power from SVD of Fourier transform")
else: print("*Using power from Fourier transform of SVD")
    

# Folder in which to save the figures
# TODO CHANGE THIS TO THE DESIRED FOLDER
fig_dst = "/projects/LEIFER/francesco/spontdyn/"

# Folders in which to export data
exp_data_folder = "/projects/LEIFER/francesco/spontdyn/exported_data/"
exp_data_folder2 = "/projects/LEIFER/francesco/spontdyn/exported_data2/"

# File containing the list of the recordings.
# TODO CHANGE THIS TO THE LOCATION OF THIS FILE ON YOUR COMPUTER
ds_list_file = "/projects/LEIFER/francesco/spontdyn_list2.txt"
tagss = ["488 AML32",
         "505 AML32",
         "488 AML70",
         "488 AKS521.1.i",
         "488 AKS522.1.i",
         "AML32H2O2 10mM"]
#group = [0,1,0,1,1,0]
group = np.arange(len(tagss))
#cs = ["C"+str(g) for g in group] # Color by group
cs = ["C"+str(i) for i in np.arange(len(tagss))] # Each tag its own color

# Convert tags to plot-friendly labels
lbls = []
for k in np.arange(len(tagss)):
    tags = tagss[k]
    tags = re.sub("AML32H2O2 10mM","505 WT\nH$_2$O$_2$",tags)
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

# Prepare figures
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(2,figsize=(12,8))
ax2 = fig2.add_subplot(111)
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
fig5 = plt.figure(5,figsize=(12,8))
ax5 = fig5.add_subplot(111)
fig6 = plt.figure(6,figsize=(12,8))
ax6 = fig6.add_subplot(111)
fig7 = plt.figure(7,figsize=(15,6))
ax7s = [fig7.add_subplot(len(tagss)//2,len(tagss)//2,k+1) for k in range(len(tagss))]
fig8 = plt.figure(8,figsize=(12,8))
ax8 = fig8.add_subplot(111)

# Array to store peak frequencies (of each neuron or PC)
maxfss = np.empty(len(np.unique(tagss)),dtype=object)
for i in np.arange(len(maxfss)): maxfss[i] = []
# Array to store a weight for the peak frequencies (e.g. singular value)
maxfss_w = np.empty(len(np.unique(tagss)),dtype=object)
for i in np.arange(len(maxfss_w)): maxfss_w[i] = []
# Array to store avg peak frequencies (of each recording)
avgmaxf = np.empty(len(np.unique(tagss)),dtype=object)
for i in np.arange(len(avgmaxf)): avgmaxf[i] = []

# Array to store fractional power in frequency band 
fracps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(fracps)): fracps[i] = np.empty(0)

# Array to store absolute power in frequency band 
ps = np.empty(len(tagss),dtype=object)
for i in np.arange(len(ps)): ps[i] = np.empty(0)

# Array to store PC absolute power in frequency band 
# up: u=singular vector, p=spectral Power
ups = np.empty(len(tagss),dtype=object)
for i in np.arange(len(ups)): ups[i] = np.empty(0)

max_fracps = np.zeros(len(fracps))
max_ups = np.zeros(len(ups))

n_neurons_pc_0 = np.empty(len(tagss),dtype=object)

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
    neurons_in_pc_ = []
    for i in np.arange(n):
        # Load files
        folder = ds_list[i]
        
        # TODO UNCOMMENT THE FOLLOWING 2 LINES AND COMMENT THE NEXT
        #sig = wormdm.signal.Signal.from_file(folder,"activity.txt",
        #                                      **signal_kwargs)
        sig = wormdm.signal.Signal.from_file(folder,"tmac",**signal_kwargs_tmac)
        
        #print("CROPPING THE RECORDING TO ONLY THE FIRST 5 MINUTES FOR MAKE THE DURATION UNIFORM FOR ALL THE RECORDINGS")
        #sig.data = sig.data[:int(5.*60./Dt)]
        
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
        # secondary range
        f0_2 = np.argmin(np.abs(f_-f_range2[0]))
        f1_2 = np.argmin(np.abs(f_-f_range2[1]))
        
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
        elif normalize_range2:
            integrand2 = ftsig[f0_2:f1_2]
            totp = np.zeros(integrand2.shape[1])
            for l in np.arange(integrand2.shape[1]):
                totp[l] = mf.num.integral(integrand2[:,l],df)
        elif no_normalize:
            totp = 1.0
        
        fracp = p/totp
        fracps[k] = np.append(fracps[k],fracp)
        
        avgft_ = np.nanmean(ftsig/totp,axis=1)
        avgft = np.interp(f,f_[f0:f1],avgft_[f0:f1])
        avgfts.append(avgft)
        
        ################
        # ACT ON THE PCS
        ################
        if use_jpca:
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
            #plt.figure(100)
            #plt.plot(u[:,0],u[:,1])
            #plt.show()
        else:
            # Compute regular SVD
            num_comp = 2
            data_ = sig.data-np.average(sig.data,axis=0)
            u,s,v = np.linalg.svd(data_,full_matrices=False)
            u *= s # denormalize
            neurons_in_pc = utilities.neurons_in_pc(v)
            
            # Compute SVD of the power spectrum
            #data2_ = np.power(np.absolute(np.fft.fft(data_,axis=0)),2)
            # What if I compute the SVD of the Fourier transform instead?
            data2_ = np.fft.fft(data_,axis=0)
            freq2 = np.fft.fftfreq(data_.shape[0],d=Dt)
            u2,s2,v2 = np.linalg.svd(data2_,full_matrices=False)
            u2 *= s2
            neurons_in_pc2 = utilities.neurons_in_pc(v2)
        
        # Compute power inside range for the PC
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
            if normalize_range2: integrand2 = np.absolute(u2[f0_2:f1_2])
            up = np.zeros(integrand.shape[1])
            for l in np.arange(integrand.shape[1]):
                up[l] = mf.num.integral(integrand[:,l],df)
                if normalize_range2:
                    up[l] /= mf.num.integral(integrand2[:,l],df)
                
        # Choose which singular vectors to use
        sel = np.arange(num_comp) 
        #try:
        #    printed
        #except:
        #    print("Using the "+str(num_comp)+" PCs with the largest power.")
        #    printed=True
        #sel = np.argsort(up)[::-1][:num_comp]
        
        ups[k] = np.append(ups[k],up[sel])        
        
        # Store peak frequency 
        maxf_this_rec = []
        for m in sel:
            # Compute the peak frequency from the quantity selected above
            # either ft_of_u or u_of_ft
            if from_ft_of_u:
                maxf = np.sum(np.absolute(ftu[f0:f1,m])*f_[f0:f1])*df
                maxf /= np.sum(np.absolute(ftu[f0:f1,m])*df)
                maxfss[k].append(maxf)
                maxf_this_rec.append(maxf)
            elif from_u_of_ft:
                u2__ = np.absolute(u2[:,m])
                #u2__[~np.logical_and(freq2>f_range[0]/1,freq2<f_range[1]*1)] = 0.
                u2__[freq2<0] = 0.0
                '''THIS IS TO FIND LOCAL MAXIMA
                u2__b = np.copy(u2__)
                u2__der = np.diff(u2__)#savgol(u2__,3,1,1)
                u2__der = np.append(u2__der,u2__der[-1])
                u2__der_s = np.sign(u2__der)
                zeros = np.append(False,np.diff(u2__der_s)<0)
                u2__[~zeros] = 0.0
                #maxf = freq2[np.argmax(u2__)]
                #maxf = np.sum(freq2[zeros]*u2__[zeros])/np.sum(u2__[zeros])'''
                maxf = np.sum(freq2*u2__)/np.sum(u2__)
                '''plt.figure(1000)
                print(tags)
                plt.plot(freq2,u2__)
                plt.axvline(maxf)
                plt.twinx()
                #plt.plot(freq2,u2__der,c="C1")
                plt.xlim(0,0.05)
                plt.show()'''
                maxfss[k].append(maxf)
                maxfss_w[k].append(s2[m]/np.sum(s2))
                maxf_this_rec.append(maxf)
        # Average of the peak frequencies of this recording
        avgmaxf[k].append(np.average(maxf_this_rec))#np.sum(maxf_this_rec*s2[sel])/np.sum(s2[sel]))####correct s2 if needed
        neurons_in_pc_.append(np.average(neurons_in_pc[:2]))
    
    max_fracps[k] = np.nanmax(fracps[k])
    max_ups[k] = np.nanmax(ups[k])
    n_neurons_pc_0[k] = neurons_in_pc_
        
    avgfts = np.array(avgfts)
    avgavgft = np.average(avgfts,axis=0)
    
    ax1.plot(f,np.convolve(avgavgft,np.ones(1),mode="same"),label=tags,c=cs[k])


# Parameters for box plots
dn = 1.0
d = 0.2
bar_width = 0.4
dy = 0.05   

# Indices of control datasets
ref_tag_i = tagss.index("488 AML32")
#test_direction = "greater"
test_direction = "less"
test_direction_ax2 = "less"

# FIG 2: Initialize plotting of significance stars and arrows
avgmaxf_ = np.empty(0)
for mfr in avgmaxf:
    avgmaxf_ = np.append(avgmaxf_,mfr)
y_sgf_ax2 = np.max(avgmaxf_)
h_ax2 = y_sgf_ax2*0.025

# FIG 5/6: Decide on which quantity to perform the significance test
# PCs
tbtested = ups
max_ups_ = np.empty(0)
for mfr in max_ups:
    max_ups_ = np.append(max_ups_,mfr)
y_sgf = np.max(max_ups_)
h = y_sgf*0.025
axtest = ax6

# individual neurons
'''
tbtested = fracps
# Starting y of the lines for significance of difference (in ax5)
max_fracps_ = np.empty(0)
for mfr in max_fracps:
    max_fracps_ = np.append(max_fracps_,mfr)
y_sgf = np.max(max_fracps_)/4
axtest = ax5
'''

for k in np.arange(len(tagss)):
    tags = tagss[k]
    tags = re.sub("AML32H2O2 10mM","505 WT\nH$_2$O$_2$",tags)
    tags = re.sub("488","488 nm\n",tags)
    tags = re.sub("505","505 nm\n",tags)
    tags = re.sub("AML32","WT",tags)
    tags = re.sub("AML70",r"$lite-1$",tags)
    tags = re.sub("AKS521.1.i",r"$gur-3$",tags)
    tags = re.sub("AKS522.1.i",r"$lite-1;gur-3$",tags)
    
    # FIG 2: PEAK FREQUENCY IN EACH RECORDING
    x = avgmaxf[k]#maxfss[k]#
    x_ref = avgmaxf[ref_tag_i]#maxfss[ref_tag_i]#
    ax2.boxplot(x,positions=[k],boxprops={"color":cs[k],"linewidth":2},
                medianprops={"color":cs[k],"linewidth":2},widths=bar_width,
                showfliers=False,whis=(0,100))
    x_scatter_2 = k*dn + mf.plt.simple_beeswarm(x)*bar_width/2
    ax2.scatter(x_scatter_2,x,edgecolor=cs[k],facecolor=cs[k],s=50,alpha=0.8,)
    
    # FIG 2: Significance of difference from 488 AML32
    if k!=ref_tag_i:
        _,pval = kstest(np.ravel(x_ref),np.ravel(x),
                        alternative=test_direction_ax2)
        stars = mf.plt.p_to_stars(pval)
            
        y0avg = np.average(x_ref)
        dyy = (np.average(x)-y0avg)/y0avg
        smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
        print("###Peak freq of",tagss[k],"is",np.around(dyy,2),smallergreater,
              "than "+tagss[ref_tag_i]+" with p",
              np.format_float_scientific(pval,1))
        
        if stars not in ["",]:
            if stars == "n.s.": stars = "p="+str(np.around(pval,2))
        
            x1 = k
            x2 = ref_tag_i
            #h_ax2 = 0.0005
            y_sgf_ax2 += 2*h_ax2
            ax2.plot([x1, x1, x2, x2], 
                     [y_sgf_ax2, y_sgf_ax2+h_ax2, y_sgf_ax2+h_ax2, y_sgf_ax2], 
                     lw=1.5, color="gray")
            ax2.plot(x1,y_sgf_ax2,'v',color="gray")
            signdyy = "+" if np.sign(dyy)==1 else "-"
            ax2.text(0.5*(x1+x2), y_sgf_ax2+h_ax2+h_ax2/5.0,
                     signdyy+str(int(abs(dyy)*100))+"% "+stars,
                     ha="center", color="gray")
    
    # FIG 3,5,6: FRACTION OF POWER FOR EACH NEURON AND PC
    y = fracps[k]
    z = ups[k]
    z2 = n_neurons_pc_0[k]
    
    ax3.bar(k*dn,np.average(y),color=cs[k],width=bar_width,alpha=0.6,label=tags)
    ax3.scatter(k*dn+np.random.random(len(fracps[k]))*bar_width/2 - bar_width/4,
                fracps[k], color=cs[k],s=0.5)
                
    # Plot histogram of the fractional powers
    ax4.hist(fracps[k],label=tags,alpha=0.2)
    
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
    
    ax7s[k].scatter(maxfss_w[k],maxfss[k],c=cs[k],label=tags)
    
    ax8.boxplot(z2,positions=[k],boxprops={"color":cs[k],"linewidth":2},
                medianprops={"color":cs[k],"linewidth":2},widths=bar_width,
                showfliers=False,whis=(0,100))
    x_scatter_2 = k*dn + mf.plt.simple_beeswarm(z2)*bar_width/2
    ax8.scatter(x_scatter_2,z2,edgecolor=cs[k],facecolor=cs[k],s=50,alpha=0.8,)
    
    # Significance of difference from 488 AML32
    if k!=ref_tag_i:
        #stats,pval = ttest_ind(np.ravel(tbtested[ref_tag_i]),np.ravel(tbtested[k]),
        #                       equal_var=False,alternative=test_direction)
        _,pval = kstest(np.ravel(tbtested[ref_tag_i]),np.ravel(tbtested[k]),
                      alternative=test_direction)
        #print("pval", pval, "stats", stats)
        stars = mf.plt.p_to_stars(pval)
            
        y0avg = np.average(tbtested[ref_tag_i])
        dyy = (np.average(tbtested[k])-y0avg)/y0avg
        smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
        print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,
              "than "+tagss[ref_tag_i]+" with p",
              np.format_float_scientific(pval,1))
        
        if stars not in ["",]:
            if stars == "n.s.": stars = "p="+str(np.around(pval,2))
            # Add significance stars to bar plot.
            #ax3.text(k, 0.35, stars)
        
            # In ax5, also add a line to show what is being compared.
            x1 = k
            x2 = ref_tag_i
            #h = 0.1*#0.05#0.0002#50
            y_sgf += 2*h
            axtest.plot([x1, x1, x2, x2], [y_sgf, y_sgf+h, y_sgf+h, y_sgf], 
                        lw=1.5, color="gray")
            axtest.plot(x1,y_sgf,'v',color="gray")
            signdyy = "+" if np.sign(dyy)==1 else "-"
            axtest.text(0.5*(x1+x2), y_sgf+h+h/5.0,
                        signdyy+str(int(abs(dyy)*100))+"% "+stars,
                        ha="center", color="gray")
            
################################################################################
################################################################################
# COMPUTE P-VALUES ALSO FOR THE RELATIVE INCREASE OF 505 WT -> 505 WT + H202
# BOTH FOR FIG5/6 AND FIG2
################################################################################
################################################################################
k = tagss.index("AML32H2O2 10mM")
k2 = tagss.index("505 AML32")

########
## FIG 2
########
test_direction = "greater"
_, pval = kstest(np.ravel(avgmaxf[k2]),np.ravel(avgmaxf[k]),
                 alternative=test_direction)
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(avgmaxf[k2])
dyy = (np.average(avgmaxf[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf_ax2 += 2*h_ax2
    ax2.plot([x1, x1, x2, x2], [y_sgf_ax2, y_sgf_ax2+h_ax2, y_sgf_ax2+h_ax2, y_sgf_ax2], lw=1.5,
             color="gray")
    ax2.plot(x1,y_sgf_ax2,'v',color="gray")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    ax2.text(0.5*(x1+x2), y_sgf_ax2+h_ax2+h_ax2/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="gray")
##########
## FIG 5/6
##########
#test_direction = "less"
#stats,pval = ttest_ind(np.ravel(fracps[k2]),np.ravel(fracps[k]),equal_var=False,
#                       alternative=test_direction)
test_direction = "greater"
_, pval = kstest(np.ravel(tbtested[k2]),np.ravel(tbtested[k]),
                 alternative=test_direction)
stars = mf.plt.p_to_stars(pval)
y0avg = np.average(tbtested[k2])
dyy = (np.average(tbtested[k])-y0avg)/y0avg
smallergreater = "smaller" if np.sign(dyy)==-1 else "greater"
print("###DF/F of",tagss[k],"is",np.around(dyy,2),smallergreater,"than "+\
      tagss[k2]+" with p",np.format_float_scientific(pval,1))
if stars not in ["",]:
    if stars == "n.s.": stars = "p="+str(np.around(pval,2))
    x1 = k
    x2 = k2
    y_sgf += 2*h
    axtest.plot([x1, x1, x2, x2], [y_sgf, y_sgf+h, y_sgf+h, y_sgf], lw=1.5,
             color="gray")
    axtest.plot(x1,y_sgf,'v',color="gray")
    signdyy = "+" if np.sign(dyy)==1 else "-"
    axtest.text(0.5*(x1+x2), y_sgf+h+h/5.0, 
                signdyy+str(int(abs(dyy)*100))+"% "+stars, 
                ha="center", color="gray")

################
################
# FINALIZE PLOTS
################
################

ax1.set_xlabel("frequency (Hz)")
ax1.set_ylabel("average power spectrum")
ax1.legend()
fig1.tight_layout()

ax2.set_xticks(np.arange(len(tagss)))
ax2.set_xticklabels(lbls)   
ax2.set_ylabel("frequency (Hz)")
ax2.legend()
fig2.tight_layout()


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
                   " Hz\nin each of the top "+str(num_comp)+" PCs",)
fig6.tight_layout()
fig6.savefig(fig_dst+"power_spectrum_bars_PCs.pdf",dpi=300,bbox_inches="tight")
fig6.savefig(fig_dst+"power_spectrum_bars_PCs.png",dpi=300,bbox_inches="tight")

ax8.set_xticks(np.arange(len(tagss)))
ax8.set_xticklabels(lbls)
ax8.spines.right.set_visible(False)
ax8.spines.top.set_visible(False)
if no_normalize:
    ax8.set_ylabel("Number of neurons in the first "+str(num_comp)+" PCs")
fig8.tight_layout()
fig8.savefig(fig_dst+"neurons_in_pc_bars_PCs.pdf",dpi=300,bbox_inches="tight")
fig8.savefig(fig_dst+"neurons_in_pc_bars_PCs.png",dpi=300,bbox_inches="tight")

for k in np.arange(len(tagss)):
    ax7s[k].axhline(f_range[0],c="k")
    ax7s[k].set_xlim(0,1.0)
    ax7s[k].set_ylim(0,0.04)
    ax7s[k].set_xlabel("FT var explained")
    ax7s[k].set_ylabel("Peak f (Hz)")
    ax7s[k].legend()
fig7.tight_layout()

plt.show()

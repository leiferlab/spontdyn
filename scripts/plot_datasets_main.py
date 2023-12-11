import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, re, sys
from sklearn.decomposition import PCA
from scipy.stats import kstest
import wormdatamodel as wormdm
import mistofrutta as mf

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)

# Hardcoding the timestep of the recording to remove the dependency from 
# additional files that should otherwise be added to the exported_data.
Dt = 1./6.

# Folder in which to save the figures
# TODO CHANGE THIS TO THE DESIRED FOLDER
fig_dst = "/projects/LEIFER/Wayan/LightEvokedDynamics/figures manuscript/"

# TODO UNCOMMENT THE VARIABLE signal_kwargs COMMENT OUT signal_kwargs_tmac
# signal_kwargs = {"preprocess": False}
signal_kwargs_tmac = {"remove_spikes": True,  "smooth": False, 
                     "nan_interp": False, "photobl_appl": False}

# TODO ADJUST TO THE CORRECT FOLDER
# Likely something like <>/exported_data2/BrainScanner<>
ds_list = ["/projects/LEIFER/PanNeuronal/20230326/BrainScanner20230326_145406/",
           "/projects/LEIFER/PanNeuronal/20230314/BrainScanner20230314_160321/",
           "/projects/LEIFER/PanNeuronal/20230404/BrainScanner20230404_182215/",
           "/projects/LEIFER/PanNeuronal/20221108/BrainScanner20221108_184729/"
           ]


tags = ["488 nm WT","505 nm WT","488 nm gur-3","505 nm wt H$_2$O$_2$"]

fig = plt.figure(1,figsize=(12,18))
hsp = 10
hsp_0 = 5#1
gs = fig.add_gridspec(hsp*len(ds_list)+hsp_0, 4)
cax = fig.add_subplot(gs[0,1])
cax2 = fig.add_subplot(gs[0,-1])

n = len(ds_list)
ax = []
axb = []
crop_t = 9*60*6
for i in np.arange(n):
    # Load files
    folder = ds_list[i]
    
    # TODO UNCOMMENT THE FOLLOWING LINE AND COMMENT THE NEXT
    #sig = wormdm.signal.Signal.from_file(folder,"activity.txt",**signal_kwargs)
    sig = wormdm.signal.Signal.from_file(folder,"tmac",**signal_kwargs_tmac)
    if sig.data.shape[0]>crop_t:
        data = sig.data[:crop_t]
    else:
        data = sig.data
    
    der = sig.get_derivative(data,39,1)[39:-39]
    
    # Run PCA
    pca = PCA()
    pcs = pca.fit_transform(der)
    weights = pca.components_
    expl_var = pca.explained_variance_ratio_
    sorter = np.argsort(weights[0])[::1]
    
    ax.append(fig.add_subplot(gs[hsp_0+hsp*i:hsp_0+hsp*(i+1),:3]))
    if i>0:
        axb.append(fig.add_subplot(gs[hsp_0+hsp*i:hsp_0+hsp*(i+1),3],projection="3d",
                                   sharex=axb[0],sharey=axb[0],sharez=axb[0]))
    else:
        axb.append(fig.add_subplot(gs[hsp_0+hsp*i:hsp_0+hsp*(i+1),3],projection="3d"))
    
    # Make colormap of the recording
    im = ax[-1].imshow(data[:,sorter].T-1.,cmap="viridis",
                       vmin=-0.8,vmax=0.8,aspect="auto")
    if i == 0:
        plt.colorbar(im,cax=cax,use_gridspec=True,orientation="horizontal")
    ax[-1].set_ylim(data.shape[1]+0.5,-0.5)
    ax[-1].set_yticks([data.shape[1],0])
    ax[-1].set_yticklabels([str(data.shape[1])+"\n","\n0"])
    
    
    # Make 3d plots of the PC
    lc = mf.plt.multicolor_3d_line(axb[-1],pcs[:,0],pcs[:,1],pcs[:,2],
                              np.arange(pcs.shape[0])*Dt,cm.gist_heat)
    if i==0:
        plt.colorbar(lc,cax=cax2,use_gridspec=True,orientation="horizontal")
    labelpad=10
    axb[-1].set_xlabel("PC1'",labelpad=labelpad,rotation=45)
    axb[-1].set_ylabel("PC2'",labelpad=labelpad,rotation=-45)
    axb[-1].set_zlabel("PC3'",labelpad=labelpad,rotation=-90)
    axb[-1].tick_params(axis='x', which='major', labelsize=12)
    axb[-1].tick_params(axis='y', which='major', labelsize=12)
    axb[-1].tick_params(axis='z', which='major', labelsize=12)
    
    if i<n-1:
        ax[-1].set_xticks([])
        
for ax_ in ax: 
    ax_.set_xlim(-0.5,9*60*6)
    #ax_.set_ylabel("Neuron")    

cax.xaxis.tick_top()
cax.set_xlabel(r"$\Delta F/F$")
cax.xaxis.set_label_position("top")
cax.set_xticks([-0.8,0,0.8])

cax2.xaxis.tick_top()
cax2.set_xlabel("t (min)")
cax2.xaxis.set_label_position("top")
xticks = np.arange(4)*3*60
cax2.set_xticks(xticks)
cax2.set_xticklabels([str(a/60) for a in xticks])

xticks = np.arange(4)*3*60
ax[-1].set_xticks(xticks*6)
ax[-1].set_xticklabels([str(a/60) for a in xticks])
ax[-1].set_xlabel("Time (min)")

axb[0].set_xlim(-0.05,0.05)
axb[0].set_ylim(-0.025,0.025)
axb[0].view_init(elev=-75,azim=135,roll=0)
axb[1].view_init(elev=-75,azim=135,roll=0)
axb[2].view_init(elev=-75,azim=135,roll=0)
axb[3].view_init(elev=-75,azim=135,roll=0)


for axb_ in axb:
    for tickl in axb_.get_xticklabels():
        tickl.set_rotation(-45)
        tickl.set_horizontalalignment("right")
        tickl.set_size(10)
    for tickl in axb_.get_yticklabels():
        tickl.set_rotation(45)
        tickl.set_horizontalalignment("left")
        tickl.set_size(10)
    for tickl in axb_.get_zticklabels():
        tickl.set_size(10)

#fig.tight_layout()
fig.savefig(fig_dst+"fig1_other.pdf",dpi=300)
fig.savefig(fig_dst+"fig1_other.png",dpi=300)
    
plt.show()

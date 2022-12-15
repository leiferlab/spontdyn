import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, re, sys
from sklearn.decomposition import PCA
from scipy.stats import kstest
import wormdatamodel as wormdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)

def multicolor(ax,x,y,z,t,c):
    points = np.array([x,y,z]).transpose().reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = Line3DCollection(segs, cmap=c)
    lc.set_array(t)
    ax.add_collection3d(lc)
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(y),np.max(y))
    ax.set_zlim(np.min(z),np.max(z))

signal_kwargs_tmac = {"remove_spikes": True,  "smooth": False, 
                     "nan_interp": True, "photobl_appl": False}


ds_list = ["/projects/LEIFER/PanNeuronal/20220112/BrainScanner20220112_193000/",
           "/projects/LEIFER/PanNeuronal/20220110/BrainScanner20220110_152224/",#"/projects/LEIFER/PanNeuronal/20220104/BrainScanner20220104_125532/",
           "/projects/LEIFER/PanNeuronal/20220415/BrainScanner20220415_151628/",
           "/projects/LEIFER/PanNeuronal/20221125/BrainScanner20221125_184646/",
           ]
tags = ["488 nm WT","505 nm WT","488 nm gur-3","505 nm wt H$_2$O$_2$"]

fig = plt.figure(1,figsize=(12,12))
hsp = 10
gs = fig.add_gridspec(hsp*len(ds_list)+1, 4)
cax = fig.add_subplot(gs[0,1])

n = len(ds_list)
ax = []
axb = []
crop_t = 10*60*6
for i in np.arange(n):
    # Load files
    folder = ds_list[i]
    rec = wormdm.data.recording(folder,legacy=True,rectype="3d",settings={"zUmOverV":200./10.})
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
    
    '''u,s,v = np.linalg.svd(der,full_matrices=False)
    u *= s # denormalize'''
    sorter = np.argsort(weights[0])[::1]
    
    ax.append(fig.add_subplot(gs[1+hsp*i:1+hsp*(i+1),:3]))
    if i>0:
        axb.append(fig.add_subplot(gs[1+hsp*i:1+hsp*(i+1),3],projection="3d",sharex=axb[0],sharey=axb[0],sharez=axb[0]))
    else:
        axb.append(fig.add_subplot(gs[1+hsp*i:1+hsp*(i+1),3],projection="3d"))
    
    # Make colormap of the recording
    im = ax[-1].imshow(data[:,sorter].T-1.,cmap="viridis",vmin=-0.8,vmax=0.8,aspect="auto")
    if i == 0:
        plt.colorbar(im,cax=cax,use_gridspec=True,orientation="horizontal")
    ax[-1].set_ylim(-0.5,data.shape[1]+0.5)
    ax[-1].set_yticks([0,data.shape[1]])
    ax[-1].set_yticklabels(["0\n","\n"+str(data.shape[1])])
    
    # Make 3d plots of the PC
    multicolor(axb[-1],pcs[:,0],pcs[:,1],pcs[:,2],np.arange(pcs.shape[0])*rec.Dt,cm.inferno)
    #multicolor(axb[-1],u[:,0],u[:,1],u[:,2],np.arange(u.shape[0])*rec.Dt,cm.viridis)
    labelpad=10
    axb[-1].set_xlabel("PC0'",labelpad=labelpad,rotation=45)
    axb[-1].set_ylabel("PC1'",labelpad=labelpad,rotation=-45)
    axb[-1].set_zlabel("PC2'",labelpad=labelpad,rotation=0)
    axb[-1].tick_params(axis='x', which='major', labelsize=12)
    axb[-1].tick_params(axis='y', which='major', labelsize=12)
    axb[-1].tick_params(axis='z', which='major', labelsize=12)
    
    if i<n-1:
        ax[-1].set_xticks([])
        
for ax_ in ax: 
    ax_.set_xlim(-0.5,10*60*6)
    ax_.set_ylabel("Neuron")    

cax.xaxis.tick_top()
cax.set_xlabel(r"$\Delta F/F$")
cax.xaxis.set_label_position("top")
cax.set_xticks([-0.8,0,0.8])

xticks = np.arange(3)*5*60
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

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/spontdyn/fig1_other.pdf",dpi=300,bbox_inches="tight")
    
plt.show()

import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, re, sys
from sklearn.decomposition import PCA
from scipy.stats import kstest
import wormdatamodel as wormdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=14)

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


ds_list_file = "/projects/LEIFER/francesco/spontdyn_list.txt"
tagss = ["488 AML32","505 AML32","AML32H2O2 10mM"]#
tagss = ["488 AML32","505 AML32","488 AML70","488 AKS521.1.i","488 AKS522.1.i","AML32H2O2 10mM"]#

max_n_ds = 0
for k in np.arange(len(tagss)):
    # Get list of recordings with given tags
    tags = tagss[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)
    max_n_ds = max(len(ds_list),max_n_ds)

fig = plt.figure(1,figsize=(20/3*len(tagss),12/13*(3*max_n_ds+1)))
hsp = 10
wsp = 3
gs = fig.add_gridspec(hsp*max_n_ds+1, wsp*len(tagss))

crop_t = 10*60*6
for k in np.arange(len(tagss)):
    ax = []
    cax = fig.add_subplot(gs[0,wsp*k+1])
    
    tags = tagss[k]
    ds_list = wormdm.signal.file.load_ds_list(ds_list_file,tags=tags,exclude_tags=None)
    n = len(ds_list)
    
    for i in np.arange(n):
        # Load files
        folder = ds_list[i]
        rec = wormdm.data.recording(folder,legacy=True,rectype="3d",settings={"zUmOverV":200./10.})
        sig = wormdm.signal.Signal.from_file(folder,"tmac",**signal_kwargs_tmac)
        if sig.data.shape[0]>crop_t:
            data = sig.data[:crop_t]
        else:
            data = sig.data
        
        ax.append(fig.add_subplot(gs[1+hsp*i:1+hsp*(i+1),k*wsp:(k+1)*wsp]))
        
        der = sig.get_derivative(data,39,1)[39:-39]
    
        # Run PCA
        pca = PCA()
        pcs = pca.fit_transform(der)
        weights = pca.components_
        expl_var = pca.explained_variance_ratio_
        sorter = np.argsort(weights[0])[::1]
        
        # Make colormap of the recording
        im = ax[-1].imshow(data[:,sorter].T-1.,cmap="viridis",vmin=-0.8,vmax=0.8,aspect="auto")
        if i == 0:
            plt.colorbar(im,cax=cax,use_gridspec=True,orientation="horizontal")

        if i<n-1:
            ax[-1].set_xticks([])
            
    for ax_ in ax: 
        ax_.set_xlim(-0.5,10*60*6)
        ax_.set_yticks([])

    cax.xaxis.tick_top()
    cax.set_xlabel(r"$\Delta F/F$")
    cax.xaxis.set_label_position("top")
    cax.set_xticks([-0.8,0,0.8])

    xticks = np.arange(3)*5*60
    ax[-1].set_xticks(xticks*6)
    ax[-1].set_xticklabels([str(a/60) for a in xticks])
    ax[-1].set_xlabel("Time (min)")

fig.suptitle("   /    ".join(tagss))
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/spontdyn/figS.pdf",dpi=300,bbox_inches="tight")
    
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA, IncrementalPCA
import fastcluster as fcl
import sys
import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib.pyplot import cm
import matplotlib as mpl
from collections import defaultdict
import os
def hex_to_rgb(_hex):
    _hex = _hex.lstrip('#')
    hlen = len(_hex)
    #print(_hex)
    return tuple(int(_hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
def heatmapper(X, xLabels=[],yLabels=[], save= os.getcwd()+os.path.sep, WRITE_CLUSTER=True, methods="tsne",CPU=os.cpu_count()//2, cluster_both=True, SHOW=True):
    """  
    X: M x N array.
    xLabels: N array. The labels or names of data X by column.  
    yLabels: M array. The labels or names of data X by row.
    save: a saving directory with a prefix
    WRITE_CLUSTER: True or False. choose if cluster information is output ot not.
    methods: "", "tsne", "umap", "pca". Dimension reduction methods to apply before hierarchical clustering.
    CPU: CPU number to use. It has effect only when tsne methods is used.
    """
    
    Xshape=np.shape(X)
    
    
    
    if WRITE_CLUSTER:
        if len(yLabels)==0:
            print("Warning: y label names are automatically set as serial numbers. Provide yLabels option so that label names make sense.")
            
            yLabels=list(map(str, range(Xshape[0])))
            #sys.exit("if WRITE_CLUSTER=True, provide xLabels")
        if cluster_both==True and len(xLabels)==0:
            print("Warning: x label names are automatically set as serial numbers. Provide xLabels option so that label names make sense.")
            xLabels=list(map(str, range(Xshape[1])))
    """
    This function generates heatmap of transcriptome data with the hierarchical clustering.  
    """
    save=save+"_"+methods
    # Compute and plot first dendrogram.
    if methods !="":
        print("reducing X axis dimension with "+methods)
        if methods=="umap":
            embeddingX = umap.UMAP(n_neighbors=5,  min_dist=0.1, metric='euclidean', n_components=2).fit_transform(X)
        elif methods=="pca":
            embeddingX = PCA(n_components=10).fit_transform(X)
        elif methods=="tsne":
            if CPU==0:
                CPU=1
            tsne = TSNE(n_jobs=CPU,perplexity = 40.000000,  n_iter=5000)
            embeddingX = tsne.fit_transform(X)
    else:
        embeddingX=np.array(X)
    fig = plt.figure(figsize=(8,20))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.8])
    print("calculating Y axis linkage")
    Y = fcl.linkage(embeddingX, method='ward', metric='euclidean')
    cmap = cm.nipy_spectral(np.linspace(0, 1, 18))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    print('drawing dendrogram...')
    Z1 = sch.dendrogram(Y, orientation='left',color_threshold=0.1*max(Y[:,2]))
    if cluster_both:
        
        Xt=np.transpose(X)
        if methods!="":
            print("reducing Y axis dimension with "+methods)
            if methods=="umap":
                
                embeddingXt = umap.UMAP(n_neighbors=5,  min_dist=0.05, metric='euclidean', n_components=2).fit_transform(Xt)
            elif methods=="pca":
                embeddingXt = PCA(n_components=10).fit_transform(Xt)
            elif methods=="tsne":
                tsne = TSNE(n_jobs=8,perplexity = 40.000000,  n_iter=5000)
                embeddingXt = tsne.fit_transform(Xt)
        else:
            embeddingXt=Xt
        ax2 = fig.add_axes([0.3,0.9,0.5,0.05])
        #Xt=np.transpose(embeddingXt)
        print("calculating X axis linkage")
        Y2 = fcl.linkage(embeddingXt, method='ward', metric='euclidean')
        
        print('drawing dendrogram...')
        cmap2 = cm.nipy_spectral(np.linspace(0, 1, 20))
        sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap2])
        Z2 = sch.dendrogram(Y2, orientation='top',color_threshold=0.2*max(Y2[:,2]))
        idx2 = Z2['leaves']
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.5,0.8])
    idx1 = Z1['leaves']
    #idx2 = Z2['leaves']
    
    X2 = X[idx1]
    if cluster_both:
        X2=X2[:,idx2]
        if WRITE_CLUSTER:
            new_xLabels=[]
            for i in idx2:
                new_xLabels.append(xLabels[i])
            cluster_list2=[]
            _tmp_set=set()
            cluster_idxs2 = defaultdict(list)
            #print(Z2['color_list'])
            for c, ic, dc in zip(Z2['color_list'], Z2['icoord'], Z2['dcoord']):
                for l in [[0, 1], [3, 2]]:
                    if dc[l[0]]==0.0:
                        i = int((ic[l[1]] - 5.0) / 10.0)
                        if not i in _tmp_set:
                            _tmp_set.add(i)
                            cluster_list2.append([i, c])
                            cluster_idxs2[c].append(i)
                        
            cluster_list2=sorted(cluster_list2)
            assert save is not ""
            with open(save+"_clusters_X_axis.txt", "w") as fo:
                #for k, v in cluster_idxs.items():
                klist=[]
                m=0
                for k, v in cluster_list2:
                    #for _v in v:
                        #print _v, idx1[_v], yLabels[idx1[_v]]
                    #print(k,v)
                    _pos=xLabels[idx2[k]]
                    #print(_pos, k, v)
                    if v=="b":
                        fo.write(_pos+"\t"+v+"\n")
                    else:
                        _key=",".join(map(str, hex_to_rgb(v)))
                        if len(klist)==0:
                            _c=";"+str(m)
                            m+=1
                        elif klist[-1] !=_key:
                            _c=";"+str(m)
                            m+=1
                        fo.write(_pos+"\t"+_key+_c+"\n")
                        klist.append(_key)
                    
                        
                        
    cluster_idxs = defaultdict(list)
    _tmp_set=set()
    cluster_list=[]
    for c, ic, dc in zip(Z1['color_list'], Z1['icoord'], Z1['dcoord']):
        for l in [[0, 1], [3, 2]]:
            if dc[l[0]]==0.0:
                i = int((ic[l[1]] - 5.0) / 10.0)
                if not i in _tmp_set:
                    _tmp_set.add(i)
                    cluster_list.append([i, c])
                    cluster_idxs[c].append(i)
                else:
                    print(c, ic, dc)
    cluster_list=sorted(cluster_list)
    if WRITE_CLUSTER:
        assert save is not ""
        with open(save+"_clusters_Y_axis.txt", "w") as fo:
            #for k, v in cluster_idxs.items():
            klist=[]
            m=0
            for k, v in cluster_list:
                #for _v in v:
                    #print _v, idx1[_v], yLabels[idx1[_v]]
                _pos=yLabels[idx1[k]]
                if v=="b":
                    fo.write(_pos+"\t"+v+"\n")
                else:
                    _key=",".join(map(str, hex_to_rgb(v)))
                    if len(klist)==0:
                        _c=";"+str(m)
                        m+=1
                    elif klist[-1] !=_key:
                        _c=";"+str(m)
                        m+=1
                    fo.write(_pos+"\t"+_key+_c+"\n")
                    klist.append(_key)
               
    
    labels = []
    sizes = []
    colors = []
    for k, v in cluster_idxs.items():
        sizes.append(len(v))

        colors.append(k)
        labels.append(len(v))
        
    sizes, colors, labels = zip(*sorted(zip(sizes, colors, labels), reverse=True))
    print("drawing heatmap")
    im = axmatrix.imshow(X2, aspect='auto', origin='lower', cmap='YlGnBu')
    if len(xLabels)<=10:
        axmatrix.set_xticks(range(len(xLabels)))
        axmatrix.set_xticklabels(xLabels, rotation=90)
    else:
        axmatrix.set_xticks([])
        axmatrix.set_xticklabels([])
    
    axmatrix.yaxis.tick_right()
    axmatrix.set_yticks([])
    axmatrix.set_yticklabels([])
    #for label in axmatrix.get_yticklabels():
        #label.set_fontname('Arial')
        #label.set_fontsize(6)
    # Plot colorbar.
    axcolor = fig.add_axes([0.5,0.05,0.16,0.02])
    pylab.colorbar(im, cax=axcolor, orientation='horizontal')
    fig2 = pylab.figure(figsize=(8,8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    if save is not "":
        
        fig.savefig(save+"_heatmap.png", format="png")
        fig2.savefig(save+"_pie.pdf", format="pdf")
    if SHOW==True:
        plt.show()
if __name__=="__main__":
    b=np.random.randint(2, size=(20,30))
    heatmapper(b)
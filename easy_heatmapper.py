import matplotlib as mpl
mpl.use("WebAgg")
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
def heatmapper(X, xLabels=[],yLabels=[], 
               save= os.getcwd()+os.path.sep, 
               WRITE_CLUSTER=True, methods="pca",
               CPU=os.cpu_count()//2, 
               cluster_both=True, 
               SHOW=True,
               tCOLOR='nipy_spectral',
               hCOLOR="YlGnBu",
               _spectral=18,
               _n_neighbors=5,
               _min_dist=0.1,
               _perplexity=50,
               _n_iter=5000,
               _pca_comp=2,
               _color_threshold=0.1):
    """  
    X: M x N array.
    xLabels: N array. The labels or names of data X by column.  
    yLabels: M array. The labels or names of data X by row.
    save: a saving directory with a prefix
    WRITE_CLUSTER: True or False. choose if cluster information is output ot not.
    methods: "", "tsne", "umap", "pca". Dimension reduction methods to apply before hierarchical clustering.
    CPU: CPU number to use. It has effect only when tsne methods is used.
    """
    plt.rcParams.update({'font.size': 12})
    Xshape=np.shape(X)
    assert len(Xshape)==2, "matrix must be two-dimensional"
    pca_comp1=Xshape[1]
    pca_comp2=Xshape[0]    
    
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
            embeddingX = umap.UMAP(n_neighbors=_n_neighbors,  min_dist=_min_dist, metric='euclidean', n_components=2).fit_transform(X)
        elif methods=="pca":
            embeddingX = PCA(n_components=_pca_comp).fit_transform(X)
        elif methods=="tsne":
            if CPU==0:
                CPU=1
            tsne = TSNE(n_jobs=CPU,perplexity = _perplexity,  n_iter=_n_iter)
            embeddingX = tsne.fit_transform(X)
        np.savez_compressed(save+"_heatmap_array.npz", X=embeddingX)
    else:
        embeddingX=np.array(X)
    fig = plt.figure(figsize=(8,20))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.8])
    print("calculating Y axis linkage")
    Y = fcl.linkage(embeddingX, method='ward', metric='euclidean')
    _cmap = cm.get_cmap(tCOLOR, _spectral)
    cmap=_cmap(range(_spectral))
    #cmap = cm.nipy_spectral(np.linspace(0, 1, _spectral))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    print('drawing dendrogram...')
    Z1 = sch.dendrogram(Y, orientation='left',color_threshold=_color_threshold*max(Y[:,2]))
    if cluster_both:
        
        Xt=np.transpose(X)
        if methods!="":
            print("reducing Y axis dimension with "+methods)
            if methods=="umap":
                
                embeddingXt = umap.UMAP(n_neighbors=_n_neighbors,  min_dist=_min_dist, metric='euclidean', n_components=2).fit_transform(Xt)
            elif methods=="pca":
                embeddingXt = PCA(n_components=_pca_comp).fit_transform(Xt)
            elif methods=="tsne":
                tsne = TSNE(n_jobs=CPU,perplexity = _perplexity,  n_iter=_n_iter)
                embeddingXt = tsne.fit_transform(Xt)
        else:
            embeddingXt=Xt
        ax2 = fig.add_axes([0.3,0.9,0.5,0.05])
        #Xt=np.transpose(embeddingXt)
        print("calculating X axis linkage")
        Y2 = fcl.linkage(embeddingXt, method='ward', metric='euclidean')
        
        print('drawing dendrogram...')
        _cmap = cm.get_cmap(tCOLOR, _spectral)
        cmap2=_cmap(range(_spectral))
        sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap2])
        Z2 = sch.dendrogram(Y2, orientation='top',color_threshold=_color_threshold*max(Y2[:,2]))
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
    im = axmatrix.imshow(X2, aspect='auto', origin='lower', cmap=hCOLOR)
    if len(xLabels)<=50:
        axmatrix.set_xticks(range(len(xLabels)))
        axmatrix.set_xticklabels(xLabels, rotation=90)
    else:
        axmatrix.set_xticks([])
        axmatrix.set_xticklabels([])
    
    axmatrix.yaxis.tick_right()
    if len(yLabels)<=50:
        axmatrix.set_yticks(range(len(yLabels)))
        axmatrix.set_yticklabels(yLabels)
    else:
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
    else:
        plt.close("all")
def scatter(X, xLabels=[],yLabels=[], 
               save= os.getcwd()+os.path.sep, 
               WRITE_CLUSTER=True, methods="tsne",
               CPU=os.cpu_count()//2, 
               SHOW=True,
               COLOR='nipy_spectral',
               _spectral=18,
               _n_neighbors=5,
               _min_dist=0.1,
               _perplexity=50,
               _n_iter=5000,
               _color_threshold=0.1,
               s=2**2,
               XX=np.array([]), color_by_value=np.array([])):
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
    yind=list(map(str, range(Xshape[0])))
    plt.rcParams.update({'font.size': 12})
    
    
    if WRITE_CLUSTER:
        if len(yLabels)==0:
            print("Warning: y label names are automatically set as serial numbers. Provide yLabels option so that label names make sense.")
            
            yLabels=list(map(str, range(Xshape[0])))
           
    save=save+"_"+methods
    # Compute and plot first dendrogram.
    if methods !="":
        print("reducing X axis dimension with "+methods)
        if methods=="umap":
            embeddingX = umap.UMAP(n_neighbors=_n_neighbors,  min_dist=_min_dist, metric='euclidean', n_components=2).fit_transform(X)
        elif methods=="pca":
            embeddingX = PCA(n_components=2).fit_transform(X)
        elif methods=="tsne":
            if CPU==0:
                CPU=1
            tsne = TSNE(n_jobs=CPU,perplexity = _perplexity,  n_iter=_n_iter)
            embeddingX = tsne.fit_transform(X)
        else:
            sys.exit("methods options can only accept umap, pca, tsne or ''.")
        np.savez_compressed(save+"_scatter_array.npz", X=embeddingX)
    else:
        print("skipping dimensionality reduction")
        if Xshape[1]!=2:
            sys.exit("if methods is '', then the shape of the matrix must be N x 2.")
        embeddingX=X
    fig, ax  = plt.subplots(figsize=(8,8))
    
    print("calculating Y axis linkage")
    Y = fcl.linkage(embeddingX, method='ward', metric='euclidean')
    _cmap = cm.get_cmap(COLOR, _spectral)
    cmap=_cmap(range(_spectral))
    sch.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
    
    print('drawing dendrogram...')
    Z1 = sch.dendrogram(Y, orientation='left',color_threshold=_color_threshold*max(Y[:,2]))

    #ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Hierarchical clustering ')
    
    
    # Plot distance matrix.
    fig2, ax2  = plt.subplots(figsize=(8,8))
    idx1 = Z1['leaves']
    #idx2 = Z2['leaves']
    
    X2 = X[idx1]
    
    cluster_idxs = defaultdict(list)
    _tmp_set=set()
    cluster_list=[]
    #print(Z1['color_list'])
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
    #print("sample num: "+str(Xshape[0])+"\ncluster_list: "+str(len(cluster_list)))
    _color_list=[""]*len(yLabels)
    if WRITE_CLUSTER:
        assert save is not ""
        with open(save+"_clusters_on_scatter_plot.txt", "w") as fo:
            #for k, v in cluster_idxs.items():
            klist=[]
            m=0
            for k, v in cluster_list:
                #for _v in v:
                    #print _v, idx1[_v], yLabels[idx1[_v]]
                _pos=str(yLabels[idx1[k]])
                _ind=str(yind[idx1[k]])
                #print(mpl.colors.hex2color(v))
                _color_list[idx1[k]]=list(mpl.colors.hex2color(v))+[1.0]
                if v=="b":
                    fo.write(_ind+"\t"+_pos+"\t"+v+"\n")
                else:
                    _key=",".join(map(str, hex_to_rgb(v)))
                    if len(klist)==0:
                        _c=";"+str(m)
                        m+=1
                    elif klist[-1] !=_key:
                        _c=";"+str(m)
                        m+=1
                    fo.write(_ind+"\t"+_pos+"\t"+_key+_c+"\n")
                    klist.append(_key)
    else:
        for k, v in cluster_list:
            
            _color_list[idx1[k]]=list(mpl.colors.hex2color(v))+[1.0]
    
    print("drawing scatter plot")
    if np.size(color_by_value)>0:
        fig3, ax3  = plt.subplots(figsize=(8,8))
        ax3.scatter(embeddingX[:, 0],embeddingX[:,1], c=color_by_value,s=s)
        fig3.savefig(save+"_scatter_goi.png", format="png")
        
    plt.scatter(embeddingX[:, 0],embeddingX[:,1], color=_color_list,s=s)
    ax2.set_title('Scatter plot colored by clusters')
    #plt.scatter(X[:, 0],X[:,1], color=_color_list)
    fig.savefig(save+"_dendro.png", format="png")
    fig2.savefig(save+"_scatter.png", format="png")
    if SHOW==True:
        plt.show()
    else:
        plt.close("all")

if __name__=="__main__":
    """
    DIM=5
    DIM2=5
    b=np.random.normal(0,1, size=(DIM,DIM2))
    for i in range(10):
        b=np.concatenate((b, np.random.normal(i/3.0, 1, size=(DIM,DIM2) )), axis=0)
    b[:]+=np.random.randint(0,5, size=(DIM2))
    np.random.shuffle(b)
    print(b)
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    #print(y)
    X=np.random.normal(0,1, size=(5,2))
    scatter(X, methods="")
    
    
    
    
    
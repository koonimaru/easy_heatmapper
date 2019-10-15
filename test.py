import numpy as np
import easy_heatmapper as eh

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
    
eh.heatmapper(X, methods="tsne")

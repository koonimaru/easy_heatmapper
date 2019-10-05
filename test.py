import numpy as np
import easy_heatmapper as eh
b=np.random.normal(0,1, size=(25,25))
    for i in range(10):
        b=np.concatenate((b, np.random.normal(i+1, 1, size=(25,25) )), axis=0)
eh.heatmapper(b)

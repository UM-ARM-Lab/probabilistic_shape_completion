
from __future__ import print_function

import IPython
import random

class UnknownSpaceSampler:
    def __init__(self, elem):
        shape = elem['known_free'].shape[1:4]        
        self.indices = [(i,j,k) for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])]
        self.indices_iter = self.indices.__iter__()
        self.pred_free = None
        self.pred_occ = None
        self.ct = 0

    def sample_conditional_unknown_space(self, model, elem, inference):

        if self.pred_free is None:
            self.pred_free = inference['predicted_free'].numpy()
            self.pred_occ = inference['predicted_occ'].numpy()


        # for ind in indices:
        ind = next(self.indices_iter)
        i = (0,ind[0],ind[1],ind[2],0)
        
        if elem['known_free'][i] == 1.0:
            return elem, inference
        if elem['known_occ'][i] == 1.0:
            return elem, inference

        if self.pred_free[i] > 0.99:
            elem['known_free'][i]=1.0
            return elem, inference
        if self.pred_occ[i] > 0.99:
            elem['known_occ'][i]=1.0
            return elem, inference

        p = self.pred_occ[i]
        if p > random.random():
            elem['known_occ'][i] = 1.0
        else:
            elem['known_free'][i] = 1.0
        

        inference = model.model(elem)
        self.pred_free = inference['predicted_free'].numpy()
        self.pred_occ = inference['predicted_occ'].numpy()
        self.ct += 1
        print("Sampling {}".format(self.ct))
        return elem, inference
        


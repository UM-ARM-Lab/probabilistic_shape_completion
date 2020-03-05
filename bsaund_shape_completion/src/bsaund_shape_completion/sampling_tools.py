
from __future__ import print_function

import tensorflow as tf
import random

import IPython

class UnknownSpaceSampler:
    def __init__(self, elem):
        shape = elem['known_free'].shape[1:4]        
        self.indices = [(i,j,k) for i in range(shape[0]) for j in range(shape[1]) for k in range(shape[2])]
        random.shuffle(self.indices)
        self.indices_iter = self.indices.__iter__()
        self.pred_free = None
        self.pred_occ = None
        self.ct = 0

    def sample(self, model, elem, inference):

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

        if self.pred_free[i] > 1.0:
            elem['known_free'][i]=1.0
            return elem, inference
        if self.pred_occ[i] > 1.0:
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

class MostConfidentSampler:
    def __init__(self, elem):
        self.shape = elem['known_free'].shape
        self.ct = 0

    def sample(self, model, elem, inference):
        po = inference['predicted_occ']
        pf = inference['predicted_free']

        unknown = 1 - (elem['known_free'] + elem['known_occ'])
        if tf.math.reduce_sum(unknown) <= 0:
            raise StopIteration
        
        c = tf.math.reduce_max(tf.abs(po - pf) * unknown)
        c = tf.math.minimum(c, 1.0)

        mask = tf.abs(po-pf) * unknown >= c
        sampled_occ = tf.random.uniform(shape=self.shape, minval=-.99, maxval=1.0) < (po-pf)
        sampled_free = tf.logical_not(sampled_occ)

        new_occ = tf.cast(tf.logical_and(mask, sampled_occ), tf.float32)
        new_free = tf.cast(tf.logical_and(mask, sampled_free), tf.float32)


        

        elem['known_occ'] += new_occ.numpy()
        elem['known_free'] += new_free.numpy()

        inference = model.model(elem)
        self.ct += 1

        new_known_ct = tf.reduce_sum(new_occ + new_free)
        
        print("Sampling iteration {}: threshold {}, new {}, still unknown {}".\
              format(self.ct, c, new_known_ct, tf.math.reduce_sum(unknown)))

        # IPython.embed()


        return elem, inference


    


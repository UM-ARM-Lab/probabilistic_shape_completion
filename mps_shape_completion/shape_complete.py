#!/usr/bin/env python
import os
import numpy as np
import tensorflow.compat.v1 as tf
import time
# config
GPU0 = '/gpu:0'
RESOLUTION = 64

class ShapeCompleter():
    def __init__(self, model_path, verbose = False):
        '''
        Constructor of the Shape_complete class. Load the model from 'model_path'.
        INPUT: verbose: print messages for debug
        '''
        tf.disable_v2_behavior()
        if not verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
            tf.logging.set_verbosity(tf.logging.FATAL)
        else:
            t_prepare_begin = time.time()

        with tf.device('/gpu:0'):
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            
            # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.saver = tf.train.import_meta_graph( model_path + 'model.cptk.meta', clear_devices=True)
            self.saver.restore(self.sess, model_path+'model.cptk')
        if verbose:
            print ('model restored!')

        self.X_occ = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        try:
            self.X_non = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
        except DataLossError:
            self.X_non = None
        self.Y_pred = tf.get_default_graph().get_tensor_by_name('aeu/Sigmoid:0')
        if verbose:
            t_prepare_end = time.time()
            print('time to initialize: {}'.format(t_prepare_end-t_prepare_begin))

    def complete(self, occ, non = None, verbose = False, save = False, id = None, out_path = None):
        '''
        Complete the 3d shape according to the occupied grids and non-occupied grids
        INPUT: occ: DATATYPE: bool. SHAPE: (64,64,64) for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids
        INPUT: non: DATATYPE: bool. SHAPE: (64,64,64), dtype: bool. for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids(64,64,64) for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids: non: (64,64,64) for a single non-occupied grids OR (batch_size,64, 64, 64) for a batch of non-occupied grids
        INPUT: verbose: DATATYPE: bool. give some messages for debug
        OUTPUT: completed shape: DATATYPE: bool. SHAPE (64,64,64) for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids
        '''
        occ, non, out_dim = self._check_input(occ, non, verbose)

        if non is None:
            y_pred = self.sess.run(self.Y_pred, feed_dict={self.X_occ:occ})
        else:
            y_pred = self.sess.run(self.Y_pred, feed_dict={self.X_occ: occ, self.X_non: non})

        # Thresholding. Threshold sets to be 0.5
        # th = 0.5
        # y_pred[y_pred >= th] = 1
        # y_pred[y_pred < th] = 0

        if out_dim == 4:
            if save:
                if id is None:
                    id = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
                for idx, y in enumerate(y_pred[:,:,:,:,0]):
                    th = 0.5
                    y[y >= th] = 1
                    y[y < th] = 0
                    self._save_grid(y, os.path.join(out_path, 'out_' + id + '_' + idx + '.binvox'), verbose)                    
            return y_pred[:,:,:,:,0]
        elif out_dim == 3:
            if save:
                if id is None:
                    id = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
                th = 0.5
                y_temp = np.copy(y_pred[0,:,:,:,0])
                y_temp[y_temp >= th] = 1
                y_temp[y_temp < th] = 0
                self._save_grid(y_temp, os.path.join(out_path, 'out_' + id + '.binvox'), verbose)
            return y_pred[0,:,:,:,0]
        else:
            raise ValueError('Internal error')

    def save_input(self, occ, non, out_path, id = None, verbose = False):
        occs, nons, out_dim = self._check_input(occ, non, verbose)
        if id is None:
            id = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        for occ, non in zip(occs[:,:,:,:,0], nons[:,:,:,:,0]):
            self._save_grid(occ, os.path.join(out_path, 'in_occ_' + id + '.binvox'), verbose)
            self._save_grid(non, os.path.join(out_path, 'in_non_' + id + '.binvox'), verbose)

    def _check_input(self, occ, non, verbose = False):
        '''
        Check the format of input. Modify them to be in batch.
        '''
        if non is None:
            if occ.ndim == 3:
                out_dim = 3
                if verbose:
                    print('Get input as single voxel')
                assert(occ.shape == (RESOLUTION,RESOLUTION,RESOLUTION))
                occ = np.expand_dims(occ, 0)
                occ = np.expand_dims(occ, 4)
            elif occ.ndim == 4:
                out_dim = 4
                if verbose:
                    print('Get input as batches. Batch size: {}'.format(occ.shape[0]))
                assert(occ.shape[-3:]==(RESOLUTION,RESOLUTION,RESOLUTION))
                occ = np.expand_dims(occ,4)
            else:
                raise ValueError('Error! Wrong dimensions!')
            return occ, None, out, dim

        if not occ.shape == non.shape:
            raise ValueError('Error! Wrong dimensions')
        if occ.ndim == 3 and non.ndim == 3:
            out_dim = 3
            if verbose:
                print('Get input as single voxel')
            assert(occ.shape == (RESOLUTION,RESOLUTION,RESOLUTION))
            assert(non.shape == (RESOLUTION,RESOLUTION,RESOLUTION))
            occ = np.expand_dims(occ,0)
            non = np.expand_dims(non,0)
            occ = np.expand_dims(occ,4)
            non = np.expand_dims(non,4)
        elif occ.ndim == 4 and non.ndim == 4:
            out_dim = 4
            if verbose:
                print('Get input as batches. Batch size: {}'.format(occ.shape[0]))
            assert(occ.shape[-3:] == (RESOLUTION,RESOLUTION,RESOLUTION))
            assert(non.shape[-3:] == (RESOLUTION,RESOLUTION,RESOLUTION))
            occ = np.expand_dims(occ,4)
            non = np.expand_dims(non,4)
        else:
            raise ValueError('Error! Wrong dimensions')
        return occ, non, out_dim

    def _save_grid(self, grids, path_and_name, verbose = False, resolution = RESOLUTION):
        import binvox_rw
        vox = binvox_rw.Voxels(grids, [RESOLUTION, RESOLUTION, RESOLUTION], [0, 0, 0], 1, 'xyz')
        with open(path_and_name, 'wb') as f:
            vox.write(f)
            if verbose:
                print('Output saved to ' + path_and_name + '.')
        
    def __del__(self):
        self.sess.close()

###DEMO###

def demo():
    '''
    demo on how to use this class.
    '''
    import binvox_rw
    # Constructor
    sc = Shape_complete(verbose=True)
    
    # Read demo binvox as (64*64*64) array
    with open('demo/occupy.binvox', 'rb') as f:
        occ = binvox_rw.read_as_3d_array(f).data
    with open('demo/non_occupy.binvox', 'rb') as f:
        non = binvox_rw.read_as_3d_array(f).data

    # Complete shape
    out = sc.complete(occ=occ,non=non,verbose=False)

    # Thresholding. Threshold sets to be 0.5
    th = 0.5
    out[out >= th] = 1
    out[out < th] = 0

    # Save to file for demo
    vox = binvox_rw.Voxels(out, [64,64,64], [0,0,0], 1, 'xyz')
    with open('demo/output.binvox','wb') as f:
        vox.write(f)
        print('Output saved to demo/output.binvox.')
        print('Please use ./viewvox demo/output.binvox to visualize the result.')

if __name__ == '__main__':
    demo()

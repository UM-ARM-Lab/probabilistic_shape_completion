

'''

Very simple neural network created by bsaund to practice coding in 
Tensorflow 2.0 (instead of 1.0)

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras.layers as tfl
import data_tools
import filepath_tools
import progressbar
import datetime
import time

import IPython


class AutoEncoder(tf.keras.Model):
    def __init__(self, params, batch_size=16):
        super(AutoEncoder, self).__init__()
        self.params = params
        self.layers_dict = {}
        self.layer_names = []
        self.setup_model()
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.001)

    def _add_layer(self, layer):
        self.layers_dict[layer.name] = layer
        self.layer_names.append(layer.name)


    def setup_model(self):

        nl = self.params['num_latent_layers']
        
        autoencoder_layers = [
            tfl.Conv3D(64, (2,2,2), padding="same",  name='conv_4_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_4_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_4_maxpool'),

            tfl.Conv3D(128, (2,2,2), padding="same", name='conv_3_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_3_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_3_maxpool'),

            tfl.Conv3D(256, (2,2,2), padding="same", name='conv_2_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_2_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_2_maxpool'),

            tfl.Conv3D(512, (2,2,2), padding="same", name='conv_1_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_1_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_1_maxpool'),

            tfl.Flatten(                             name='flatten'),

            tfl.Dense(nl, activation='relu',         name='latent'),
            
            tfl.Dense(32768, activation='relu',      name='expand'),
            tfl.Reshape((4,4,4,512),                 name='reshape'),
            

            tfl.Conv3DTranspose(256, (2,2,2,), strides=2, name='deconv_1_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_1_activation'),
            tfl.Conv3DTranspose(128, (2,2,2,), strides=2, name='deconv_2_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_2_activation'),
            tfl.Conv3DTranspose(64, (2,2,2,), strides=2,  name='deconv_3_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_3_activation'),
            tfl.Conv3DTranspose(2, (2,2,2,), strides=2,   name='deconv_4_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_4_activation')
        ]
        if self.params['is_u_connected'] and self.params['use_final_unet_layer']:
            extra_unet_layers = [
                tfl.Conv3D(2, (1,1,1,), use_bias=False,                  name='unet_combine'),
                # tfl.Activation(tf.nn.relu,                             name='unet_final_activation'),
            ]
            if self.params['final_activation'] == 'sigmoid':
                extra_unet_layers.append(tfl.Activation(tf.math.sigmoid, name='unet_final_activation'))
            if self.params['final_activation'] == 'relu':
                extra_unet_layers.append(tfl.Activation(tf.nn.relu,      name='unet_final_activation'))
            

            autoencoder_layers = autoencoder_layers + extra_unet_layers

        for l in autoencoder_layers:
            self._add_layer(l)






    def call(self, inputs, training=False):
        known_occ = inputs['known_occ']
        known_free = inputs['known_free']

        unet = self.params['is_u_connected']

        x = tfl.concatenate([known_occ, known_free], axis=4)


        u5 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u5')(x, training=training)
        x = self.layers_dict['conv_4_conv'](x)
        x = self.layers_dict['conv_4_activation'](x)
        x = self.layers_dict['conv_4_maxpool'](x)
        u4 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u4')(x, training=training)
        x = self.layers_dict['conv_3_conv'](x)
        x = self.layers_dict['conv_3_activation'](x)
        x = self.layers_dict['conv_3_maxpool'](x)
        u3 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u3')(x, training=training)
        x = self.layers_dict['conv_2_conv'](x)
        x = self.layers_dict['conv_2_activation'](x)
        x = self.layers_dict['conv_2_maxpool'](x)
        u2 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u2')(x, training=training)
        x = self.layers_dict['conv_1_conv'](x)
        x = self.layers_dict['conv_1_activation'](x)
        x = self.layers_dict['conv_1_maxpool'](x)
        u1 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u1')(x, training=training)
            
        x = self.layers_dict['flatten'](x)
        x = self.layers_dict['latent'](x)
        x = self.layers_dict['expand'](x)
        x = self.layers_dict['reshape'](x)

        if(unet):
            x = tfl.concatenate([x, u1], axis=4, name='u_1')
        x = self.layers_dict['deconv_1_deconv'](x)
        x = self.layers_dict['deconv_1_activation'](x)
        if(unet):
            x = tfl.concatenate([x, u2], axis=4, name='u_2')
        x = self.layers_dict['deconv_2_deconv'](x)
        x = self.layers_dict['deconv_2_activation'](x)
        if(unet):
            x = tfl.concatenate([x, u3], axis=4, name='u_3')
        x = self.layers_dict['deconv_3_deconv'](x)
        x = self.layers_dict['deconv_3_activation'](x)
        if(unet):
            x = tfl.concatenate([x, u4], axis=4, name='u_4')
        x = self.layers_dict['deconv_4_deconv'](x)
        x = self.layers_dict['deconv_4_activation'](x)
        if(unet and self.params['use_final_unet_layer']):
            x = tfl.concatenate([x, u5], axis=4, name='u_5')
            x = self.layers_dict['unet_combine'](x)
            x = self.layers_dict['unet_final_activation'](x)

        

        occ, free = tf.split(x, 2, axis=4)
        
        return {'predicted_occ':occ, 'predicted_free':free}

    @tf.function
    def mse_loss(self, metrics):
        l_occ = tf.reduce_sum(metrics['mse/occ']) * (1.0/self.batch_size)
        l_free = tf.reduce_sum(metrics['mse/free']) * (1.0/self.batch_size)
        return l_occ + l_free


    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)
            
        
        def step_fn(batch):
            with tf.GradientTape() as tape:
                output = self(batch, training=True)
                # loss = tf.reduce_mean(tf.abs(output - example['gt']))
                # loss = tf.reduce_mean(tf.mse(output - example['gt']))
                # mse = tf.keras.losses.MSE(batch['gt'], output)

                # mse_occ = tf.losses.mean_squared_error(batch['gt_occ'], output['predicted_occ'])
                acc_occ = tf.math.abs(batch['gt_occ'] - output['predicted_occ'])
                mse_occ = tf.math.square(acc_occ)
                # mse_free = tf.losses.mean_squared_error(batch['gt_free'], output['predicted_free'])
                acc_free = tf.math.abs(batch['gt_free'] - output['predicted_free'])
                mse_free = tf.math.square(acc_free)

                unknown_occ = batch['gt_occ'] - batch['known_occ']
                unknown_free = batch['gt_free'] - batch['known_free']
                
                metrics = {"mse/occ": mse_occ, "acc/occ": acc_occ,
                           "mse/free": mse_free, "acc/free": acc_free,
                           "pred|gt/p(predicted_occ|gt_occ)": p_x_given_y(output['predicted_occ'],
                                                                  batch['gt_occ']),
                           "pred|gt/p(predicted_free|gt_free)": p_x_given_y(output['predicted_free'],
                                                                    batch['gt_free']),
                           "pred|known/p(predicted_occ|known_occ)": p_x_given_y(output['predicted_occ'],
                                                                                batch['known_occ']),
                           "pred|known/p(predicted_free|known_free)": p_x_given_y(output['predicted_free'],
                                                                                  batch['known_free']),
                           "pred|gt/p(predicted_occ|gt_free)": p_x_given_y(output['predicted_occ'],
                                                                           batch['gt_free']),
                           "pred|gt/p(predicted_free|gt_occ)": p_x_given_y(output['predicted_free'],
                                                                           batch['gt_occ']),
                           "pred|known/p(predicted_occ|known_free)": p_x_given_y(output['predicted_occ'],
                                                                                 batch['known_free']),
                           "pred|known/p(predicted_free|known_occ)": p_x_given_y(output['predicted_free'],
                                                                                 batch['known_occ']),
                           "pred|unknown/p(predicted_occ|unknown_occ)": p_x_given_y(output['predicted_occ'],
                                                                                    unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_occ)": p_x_given_y(output['predicted_free'],
                                                                                     unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_free)": p_x_given_y(output['predicted_free'],
                                                                                      unknown_free),
                           "pred|unknown/p(predicted_occ|unknown_free)": p_x_given_y(output['predicted_occ'],
                                                                                      unknown_free),
                           "sanity/p(gt_occ|known_occ)": p_x_given_y(batch['gt_occ'], batch['known_occ']),
                           "sanity/p(gt_free|known_occ)": p_x_given_y(batch['gt_free'], batch['known_occ']),
                           "sanity/p(gt_occ|known_free)": p_x_given_y(batch['gt_occ'], batch['known_free']),
                           "sanity/p(gt_free|known_free)": p_x_given_y(batch['gt_free'], batch['known_free']),
                           }
                
                loss = self.mse_loss(metrics)
                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)


                self.opt.apply_gradients(list(zip(gradients, variables)))
                metrics.update(self.get_insights(variables, gradients))
                return loss, metrics
            
        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m

    @tf.function
    def get_insights(self, variables, gradients):
        final_conv = variables[-1]
        final_grad = gradients[-1]
        insights = {}
        if self.params['is_u_connected'] and self.params['use_final_unet_layer']:
            
            unet_insights = {
                "weights/know_occ->pred_occ": final_conv[-1][0,0,2,0],
                "weights/know_occ->pred_free": final_conv[-1][0,0,2,1],
                "weights/know_free->pred_occ": final_conv[-1][0,0,3,0],
                "weights/know_free->pred_free": final_conv[-1][0,0,3,1],
                "gradients/know_occ->pred_occ": final_grad[-1][0,0,2,0],
                "gradients/know_occ->pred_free": final_grad[-1][0,0,2,1],
                "gradients/know_free->pred_occ": final_grad[-1][0,0,3,0],
                "gradients/know_free->pred_free": final_grad[-1][0,0,3,1]
            }
            insights.update(unet_insights)
        return insights




@tf.function
def p_x_given_y(x, y):
    """
    Returns the reduce p(x|y)
    Clips x from 0 to one, then filters and normalizes by y
    Assumes y is a tensor where every element is 0.0 or 1.0
    """
    clipped = tf.clip_by_value(x, 0.0, 1.0)
    return tf.reduce_sum(clipped * y) / tf.reduce_sum(y)










class Dropout(tf.keras.layers.Layer):
    def __init__(self, name="dropout"):
        super(Dropout, self).__init__(name=name)

    @tf.function
    def call(self, x, keep_prob, training=False):
        if training:
            x = x * tf.cast(tf.random.uniform(x.shape) < keep_prob, tf.float32)
        return x

    
class MaskedConv3D(tf.keras.layers.Layer):
    def __init__(self, conv_size, in_channels, out_channels, name='masked_conv_3d', is_first_layer=False):
        super(MaskedConv3D, self).__init__(name=name)
        self.conv_size = conv_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_first_layer = is_first_layer

    def build(self, input_shape):
        conv_vars = int(self.conv_size)**3

        if self.is_first_layer:
            n_train = (conv_vars/2) * self.in_channels * self.out_channels
            n_zeros = (conv_vars/2+1) * self.in_channels * self.out_channels
        else:
            n_zeros = (conv_vars/2) * self.in_channels * self.out_channels
            n_train = (conv_vars/2+1) * self.in_channels * self.out_channels

        
        self.a = self.add_weight(name=self.name + 'trainable',
                                 shape=[n_train],
                                 initializer=tf.initializers.GlorotUniform(),
                                 trainable=True)
        self.b = self.add_weight(name=self.name + 'masked',
                                 shape=[n_zeros],
                                 initializer='zeros',
                                 trainable=False)
    @tf.function
    def call(self, inputs):
        cs = self.conv_size
        f = tf.reshape(tf.concat([self.a,self.b], axis=0),
                       [cs, cs, cs, self.in_channels, self.out_channels])

        return tf.nn.conv3d(inputs, f, strides=[1,1,1,1,1], padding='SAME')

class VoxelCNN(tf.keras.Model):
    def __init__(self, params, batch_size=16):
        super(VoxelCNN, self).__init__()
        self.params = params
        self.layers_dict = {}
        self.layer_names = []
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.001)
        self.setup_model()

    def setup_model(self):
        conv_size = 3

        self.conv_layers = [
            MaskedConv3D(conv_size, 1, 16,      name='masked_conv_1', is_first_layer=True),
            tfl.Activation(tf.nn.elu,           name='conv_1_activation'),
            MaskedConv3D(conv_size, 16, 32,     name='masked_conv_2'),
            tfl.Activation(tf.nn.elu,           name='conv_2_activation'),
            MaskedConv3D(conv_size, 32, 64,     name='masked_conv_3'),
            tfl.Activation(tf.nn.elu,           name='conv_3_activation'),
            # MaskedConv3D(conv_size, 64, 128,  name='masked_conv_4'),
            # tfl.Activation(tf.nn.elu,           name='conv_4_activation'),
            # MaskedConv3D(conv_size, 128, 64,  name='masked_conv_5'),
            # tfl.Activation(tf.nn.elu,           name='conv_5_activation'),
            MaskedConv3D(conv_size, 64, 32,     name='masked_conv_6'),
            tfl.Activation(tf.nn.elu,           name='conv_6_activation'),
            MaskedConv3D(conv_size, 32, 16,     name='masked_conv_7'),
            tfl.Activation(tf.nn.elu,           name='conv_7_activation'),
            MaskedConv3D(conv_size, 16, 1,      name='masked_conv_8'),
            # 
            ]
        if self.params['final_activation'] == 'sigmoid':
            self.conv_layers.append(tfl.Activation(tf.nn.sigmoid,       name='conv_8_activation'))
        elif self.params['final_activation'] == 'elu':
            self.conv_layers.append(tfl.Activation(tf.nn.elu,           name='conv_8_activation'))
        
        # for l in conv_layers:
        #     self._add_layer(l)

    @tf.function
    def call(self, inputs, training = False):
        if training:
            x = inputs['gt_occ']
            x = Dropout()(x, keep_prob=tf.random.uniform([1]), training=True)
        else:
            x = inputs['conditioned_occ']

        for l in self.conv_layers:
            x = l(x)
        return {'predicted_occ':x, 'predicted_free':1.0-x}


    @tf.function
    def mse_loss(self, metrics):
        l_occ = tf.reduce_sum(metrics['mse/occ']) * (1.0/self.batch_size)
        return l_occ


    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)
            
        
        def step_fn(batch):
            with tf.GradientTape() as tape:
                output = self(batch, training=True)
                acc_occ = tf.math.abs(batch['gt_occ'] - output['predicted_occ'])
                mse_occ = tf.math.square(acc_occ)
                acc_free = tf.math.abs(batch['gt_free'] - output['predicted_free'])
                mse_free = tf.math.square(acc_free)

                unknown_occ = batch['gt_occ'] - batch['known_occ']
                unknown_free = batch['gt_free'] - batch['known_free']
                
                metrics = {"mse/occ": mse_occ, "acc/occ": acc_occ,
                           "mse/free": mse_free, "acc/free": acc_free,
                           "pred|gt/p(predicted_occ|gt_occ)": p_x_given_y(output['predicted_occ'],
                                                                  batch['gt_occ']),
                           "pred|gt/p(predicted_free|gt_free)": p_x_given_y(output['predicted_free'],
                                                                    batch['gt_free']),
                           "pred|known/p(predicted_occ|known_occ)": p_x_given_y(output['predicted_occ'],
                                                                                batch['known_occ']),
                           "pred|known/p(predicted_free|known_free)": p_x_given_y(output['predicted_free'],
                                                                                  batch['known_free']),
                           "pred|gt/p(predicted_occ|gt_free)": p_x_given_y(output['predicted_occ'],
                                                                           batch['gt_free']),
                           "pred|gt/p(predicted_free|gt_occ)": p_x_given_y(output['predicted_free'],
                                                                           batch['gt_occ']),
                           "pred|known/p(predicted_occ|known_free)": p_x_given_y(output['predicted_occ'],
                                                                                 batch['known_free']),
                           "pred|known/p(predicted_free|known_occ)": p_x_given_y(output['predicted_free'],
                                                                                 batch['known_occ']),
                           "pred|unknown/p(predicted_occ|unknown_occ)": p_x_given_y(output['predicted_occ'],
                                                                                    unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_occ)": p_x_given_y(output['predicted_free'],
                                                                                     unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_free)": p_x_given_y(output['predicted_free'],
                                                                                      unknown_free),
                           "pred|unknown/p(predicted_occ|unknown_free)": p_x_given_y(output['predicted_occ'],
                                                                                      unknown_free),
                           "sanity/p(gt_occ|known_occ)": p_x_given_y(batch['gt_occ'], batch['known_occ']),
                           "sanity/p(gt_free|known_occ)": p_x_given_y(batch['gt_free'], batch['known_occ']),
                           "sanity/p(gt_occ|known_free)": p_x_given_y(batch['gt_occ'], batch['known_free']),
                           "sanity/p(gt_free|known_free)": p_x_given_y(batch['gt_free'], batch['known_free']),
                           }
                
                loss = self.mse_loss(metrics)
                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)

                self.opt.apply_gradients(list(zip(gradients, variables)))
                return loss, metrics
            
        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m








    
class Network:
    def __init__(self, params=None, trial_name=None):
        self.batch_size = 16
        self.side_length = 64
        self.num_voxels = self.side_length ** 3

        file_fp = os.path.dirname(__file__)
        fp = filepath_tools.get_trial_directory(os.path.join(file_fp, "../trials/"),
                                                expect_reuse = (params is None),
                                                nick=trial_name)
        self.params = filepath_tools.handle_params(file_fp, fp, params)

        self.checkpoint_path = os.path.join(fp, "training_checkpoints/")

        train_log_dir = os.path.join(fp, 'logs/train')
        test_log_dir = os.path.join(fp, 'logs/test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        if self.params['network'] == 'VoxelCNN':
            self.model = VoxelCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'AutoEncoder':
            self.model = AutoEncoder(self.params, batch_size=self.batch_size)
        else:
            raise Exception('Unknown Model Type')

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        epoch=tf.Variable(0),
                                        train_time=tf.Variable(0.0),
                                        optimizer=self.model.opt, net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=1)

        self.num_batches = None
        self.restore()
        

    def restore(self):
        status = self.ckpt.restore(self.manager.latest_checkpoint)

        # Suppress warning 
        if self.manager.latest_checkpoint:
            status.assert_existing_objects_matched()

    def count_params(self):
        self.model.summary()

    def build_model(self, dataset):
        # self.model.evaluate(dataset.take(16))
        elem = next(dataset.take(self.batch_size).batch(self.batch_size).__iter__())
        elem['conditioned_occ'] = elem['known_occ']
        tf.summary.trace_on(graph=True, profiler=False)
        self.model(elem)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.ckpt.step.numpy())




    def write_summary(self, summary_dict):
        with self.train_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())


    def train_batch(self, dataset):
        if self.num_batches is not None:
            max_size = str(self.num_batches)
        else:
            max_size = '???'
        
        widgets=[
            '  ', progressbar.Counter(), '/', max_size,
            ' ', progressbar.Variable("Loss"), ' ',
            progressbar.Bar(),
            ' [', progressbar.Variable("TrainTime"), '] ',
            ' (', progressbar.ETA(), ') ',
            ]


        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_batches) as bar:
            self.num_batches = 0
            t0 = time.time()
            for batch in dataset:
                self.num_batches+=1
                self.ckpt.step.assign_add(1)
                
                ret = self.model.train_step(batch)
                time_str = str(datetime.timedelta(seconds=int(self.ckpt.train_time.numpy())))
                bar.update(self.num_batches, Loss=ret['loss'].numpy(),
                           TrainTime=time_str)
                self.write_summary(ret)
                self.ckpt.train_time.assign_add(time.time() - t0)
                t0 = time.time()

        
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        print("loss {:1.3f}".format(ret['loss'].numpy()))
        
    def train(self, dataset):
        self.build_model(dataset)
        self.count_params()
        # dataset = dataset.shuffle(10000)


        # batched_ds = dataset.batch(self.batch_size, drop_remainder=True).prefetch(64)
        batched_ds = dataset.batch(self.batch_size).prefetch(64)
        
        num_epochs = 1000
        while self.ckpt.epoch < num_epochs:
            self.ckpt.epoch.assign_add(1)
            print('')
            print('==  Epoch {}/{}  '.format(self.ckpt.epoch.numpy(), num_epochs) + '='*65)
            self.train_batch(batched_ds)
            print('='*80)
        
        

    def train_and_test(self, dataset):
        train_ds = dataset

        if self.params['simulate_partial_completion']:
            train_ds = data_tools.simulate_partial_completion(train_ds)
        if self.params['simulate_random_partial_completion']:
            train_ds = data_tools.simulate_random_partial_completion(train_ds)
        
        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))
        



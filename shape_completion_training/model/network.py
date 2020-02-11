

'''

Very simple neural network created by bsaund to practice coding in 
Tensorflow 2.0 (instead of 1.0)

'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import data_tools
import filepath_tools
import progressbar

import IPython


class AutoEncoder(tf.keras.Model):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()
        self.params = params
        self.setup_model()


    def setup_model(self):
        ip = (64, 64, 64, 2)
        self.autoencoder_layers = [
            tf.keras.layers.Conv3D(64, (2,2,2), input_shape=ip, padding="same"),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.MaxPool3D((2,2,2)),

            tf.keras.layers.Conv3D(128, (2,2,2), padding="same"),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.MaxPool3D((2,2,2)),

            tf.keras.layers.Conv3D(256, (2,2,2), padding="same"),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.MaxPool3D((2,2,2)),

            tf.keras.layers.Conv3D(512, (2,2,2), padding="same"),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.MaxPool3D((2,2,2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.params['num_latent_layers'], activation='relu'),
            
            tf.keras.layers.Dense(32768, activation='relu'),
            tf.keras.layers.Reshape((4,4,4,512)),
            

            tf.keras.layers.Conv3DTranspose(256, (2,2,2,), strides=2),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv3DTranspose(128, (2,2,2,), strides=2),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv3DTranspose(64, (2,2,2,), strides=2),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv3DTranspose(2, (2,2,2,), strides=2)
        ]


    def call(self, inputs):
        known_occ = inputs['known_occ']
        known_free = inputs['known_free']

        x = tf.keras.layers.concatenate([known_occ, known_free], axis=4)

        for layer in self.autoencoder_layers:
            x = layer(x)

        occ, free = tf.split(x, 2, axis=4)
        
        return {'predicted_occ':occ, 'predicted_free':free}



class AutoEncoderWrapper:
    def __init__(self, params=None):
        self.batch_size = 16
        self.side_length = 64
        self.num_voxels = self.side_length ** 3

        expect_load_from_file = (params is None)
        fp = filepath_tools.get_trial_directory(os.path.join(os.path.dirname(__file__), "../trials/"),
                                                expect_reuse = expect_load_from_file)
        if expect_load_from_file:
            defaults_fp = os.path.join(os.path.dirname(__file__), "../model/")
            params = filepath_tools.load_params(defaults_fp, fp)
        else:
            filepath_tools.write_params(fp, params)

        self.checkpoint_path = os.path.join(fp, "training_checkpoints/")

        train_log_dir = os.path.join(fp, 'logs/train')
        test_log_dir = os.path.join(fp, 'logs/test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.model = AutoEncoder(params)
            self.opt = tf.keras.optimizers.Adam(0.001)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.opt, net=self.model)
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
        self.model.predict(dataset.take(self.batch_size).batch(self.batch_size))

    @tf.function
    def mse_loss(self, metrics):
        l_occ = tf.reduce_sum(metrics['mse_occ']) * (1.0 / self.batch_size)
        l_free = tf.reduce_sum(metrics['mse_free']) * (1.0 / self.batch_size)
        return l_occ + l_free


    @tf.function
    def train_step(self, batch):
        def reduce_from_gpu(val):
            v2 = self.strategy.reduce(tf.distribute.ReduceOp.SUM, val, axis=0)
            return tf.reduce_mean(v2) * (1.0 / self.batch_size)
            
        
        def step_fn(batch):
            with tf.GradientTape() as tape:
                output = self.model(batch)
                # loss = tf.reduce_mean(tf.abs(output - example['gt']))
                # loss = tf.reduce_mean(tf.mse(output - example['gt']))
                # mse = tf.keras.losses.MSE(batch['gt'], output)
                mse_occ = tf.losses.mean_squared_error(batch['gt_occ'], output['predicted_occ'])
                acc_occ = tf.abs(batch['gt_occ'] - output['predicted_occ'])
                mse_free = tf.losses.mean_squared_error(batch['gt_free'], output['predicted_free'])
                acc_free = tf.abs(batch['gt_free'] - output['predicted_free'])
                
                metrics = {"mse_occ": mse_occ, "acc_occ": acc_occ,
                           "mse_free": mse_free, "acc_free": acc_free}
                
                # loss = tf.reduce_sum(mse_occ) * (1.0 / self.batch_size)
                loss = self.mse_loss(metrics)
                
                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)
                self.opt.apply_gradients(list(zip(gradients, variables)))
                return loss, metrics
            
        loss, metrics = self.strategy.experimental_run_v2(step_fn, args=(batch, ))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        m = {k: reduce_from_gpu(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m


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
            ' [', progressbar.Timer(), '] ',
            ' (', progressbar.ETA(), ') ',
            ]


        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_batches) as bar:
            self.num_batches = 0
            for batch in dataset:
                self.num_batches+=1
                self.ckpt.step.assign_add(1)
                
                ret = self.train_step(batch)
                bar.update(self.num_batches, Loss=ret['loss'].numpy())
                self.write_summary(ret)

        
        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        print("loss {:1.3f}".format(ret['loss'].numpy()))
        
    def train(self, dataset):
        self.build_model(dataset)
        self.count_params()
        dataset.shuffle(10000)

        batched_ds = dataset.batch(self.batch_size, drop_remainder=True)
        dist_ds = self.strategy.experimental_distribute_dataset(batched_ds)
        
        num_epochs = 1000
        for i in range(num_epochs):
            print('')
            print('==  Epoch {}/{}  '.format(i+1, num_epochs) + '='*65)
            self.train_batch(dist_ds)
            print('='*80)
        
        

    def train_and_test(self, dataset):
        train_ds = dataset
        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))
        



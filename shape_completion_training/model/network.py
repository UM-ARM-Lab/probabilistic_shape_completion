

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
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.setup_model()

    def setup_model(self):
        # self.flatten = tf.keras.layers.Flatten()
        # self.unflatten = tf.keras.layers.Reshape((64, 64, 64, 1))

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
            tf.keras.layers.Dense(200, activation='relu'),
            
            tf.keras.layers.Dense(32768, activation='relu'),
            tf.keras.layers.Reshape((4,4,4,512)),
            

            tf.keras.layers.Conv3DTranspose(256, (2,2,2,), strides=2),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv3DTranspose(128, (2,2,2,), strides=2),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv3DTranspose(64, (2,2,2,), strides=2),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv3DTranspose(1, (2,2,2,), strides=2)
        ]


    def call(self, inputs):
        known_occ = inputs['known_occ']
        known_free = inputs['known_free']

        x = tf.keras.layers.concatenate([known_occ, known_free], axis=4)

        for layer in self.autoencoder_layers:
            x = layer(x)
        
        return x



class AutoEncoderWrapper:
    def __init__(self):
        self.batch_size = 16
        self.side_length = 64
        self.num_voxels = self.side_length ** 3

        fp = filepath_tools.get_trial_directory(os.path.join(os.path.dirname(__file__), "../trials/"))

        self.checkpoint_path = os.path.join(fp, "training_checkpoints/")

        train_log_dir = os.path.join(fp, 'logs/train')
        test_log_dir = os.path.join(fp, 'logs/test')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        # self.restore_path = os.path.join(os.path.dirname(__file__), "../restore/cp.ckpt")

        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.model = AutoEncoder()
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
        # tots = len(tf.training_variables())
        # print("There are " + str(tots) + " training variables")
        self.model.summary()

    def build_model(self, dataset):
        # self.model.evaluate(dataset.take(16))
        self.model.predict(dataset.take(self.batch_size).batch(self.batch_size))

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
                mse = tf.losses.mean_squared_error(batch['gt'], output)
                acc = tf.abs(batch['gt']- output)
                loss = tf.reduce_sum(mse) * (1.0 / self.batch_size)
                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)
                self.opt.apply_gradients(list(zip(gradients, variables)))
                return mse, acc
        mse, acc = self.strategy.experimental_run_v2(step_fn, args=(batch, ))
        mse = reduce_from_gpu(mse)
        acc = reduce_from_gpu(acc)
        return {'mse':mse, 'acc':acc, 'loss':mse}


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
        # train_ds = dataset.repeat(10)
        # train_ds = dataset.skip(100)
        # test_ds = dataset.take(100)
        # IPython.embed()
        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))
        



if __name__ == "__main__":
    print("hi")
    sn = SimpleNetwork()
    # sn.simple_pass()
    sn.forward_model()

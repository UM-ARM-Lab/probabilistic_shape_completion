from shape_completion_training.model import utils
from shape_completion_training.model.normalizing_ae import NormalizingAE

utils.set_gpu_with_lowest_memory()
import tensorflow as tf
from shape_completion_training.model import filepath_tools
from shape_completion_training.model.other_model_architectures.auto_encoder import AutoEncoder
from shape_completion_training.model.other_model_architectures.augmented_ae import Augmented_VAE
# from voxelcnn import VoxelCNN, StackedVoxelCNN
from shape_completion_training.model.other_model_architectures.voxelcnn import VoxelCNN
from shape_completion_training.model.baselines.vae import VAE, VAE_GAN
from shape_completion_training.model.other_model_architectures.conditional_vcnn import ConditionalVCNN
from shape_completion_training.model.other_model_architectures.ae_vcnn import AE_VCNN
from shape_completion_training.model.baselines.three_D_rec_gan import ThreeD_rec_gan
from shape_completion_training.model.flow import RealNVP
import progressbar
import datetime
import time


class ModelRunner:
    def __init__(self, training, group_name=None, trial_path=None, params=None, write_summary=True):
        """
        @type training: bool
        @param training: 
        @param group_name: 
        @param trial_path: 
        @param params: 
        @param write_summary: 
        """

        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.side_length = 64
        self.num_voxels = self.side_length ** 3
        self.training = training

        self.trial_path, self.params = filepath_tools.create_or_load_trial(group_name=group_name,
                                                                           params=params,
                                                                           trial_path=trial_path,
                                                                           write_summary=write_summary)
        self.group_name = self.trial_path.parts[-2]

        self.batch_size = 1 if not self.training else params['batch_size']

        self.train_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/train").as_posix())
        self.test_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/test").as_posix())

        if self.params['network'] == 'VoxelCNN':
            self.model = VoxelCNN(self.params, batch_size=self.batch_size)
        # if self.params['network'] == 'StackedVoxelCNN':
        #     self.model = StackedVoxelCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'AutoEncoder':
            self.model = AutoEncoder(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'VAE':
            self.model = VAE(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'VAE_GAN':
            self.model = VAE_GAN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'Augmented_VAE':
            self.model = Augmented_VAE(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'Conditional_VCNN':
            self.model = ConditionalVCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'AE_VCNN':
            self.model = AE_VCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == "RealNVP":
            self.model = RealNVP(hparams=self.params, batch_size=self.batch_size, training=training)
        elif self.params['network'] == "NormalizingAE":
            self.model = NormalizingAE(self.params, batch_size=self.batch_size)
            self.model.flow = ModelRunner(training=False, trial_path=self.params['flow']).model.flow
        elif self.params['network'] == "3D_rec_gan":
            self.model = ThreeD_rec_gan(self.params, batch_size=self.batch_size)
        else:
            raise Exception('Unknown Model Type')

        self.num_batches = None

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        epoch=tf.Variable(0),
                                        train_time=tf.Variable(0.0),
                                        optimizer=self.model.optimizer, net=self.model)
        self.checkpoint_path = self.trial_path / "training_checkpoints/"
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path.as_posix(), max_to_keep=1)
        self.restore()

    def restore(self):
        status = self.ckpt.restore(self.manager.latest_checkpoint)

        # Suppress warning 
        if self.manager.latest_checkpoint:
            status.assert_existing_objects_matched()

    def count_params(self):
        self.model.summary()

    def build_model(self, dataset):
        elem = dataset.take(self.batch_size).batch(self.batch_size)
        tf.summary.trace_on(graph=True, profiler=False)
        self.model.predict(elem)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.ckpt.step.numpy())

        # tf.keras.utils.plot_model(self.model, (self.trial_path / 'network.png').as_posix(),
        #                           show_shapes=True)

    def write_summary(self, summary_dict):
        with self.train_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())

    def train_batch(self, dataset, summary_period=10):
        if self.num_batches is not None:
            max_size = str(self.num_batches)
        else:
            max_size = '???'

        widgets = [
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
                self.num_batches += 1
                self.ckpt.step.assign_add(1)

                _, ret = self.model.train_step(batch)
                time_str = str(datetime.timedelta(seconds=int(self.ckpt.train_time.numpy())))
                bar.update(self.num_batches, Loss=ret['loss'].numpy(),
                           TrainTime=time_str)
                if self.num_batches % summary_period == 0:
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
            print('==  Epoch {}/{}  '.format(self.ckpt.epoch.numpy(), num_epochs) + '=' * 25
                  + ' ' + self.group_name + ' ' + '=' * 20)
            self.train_batch(batched_ds)
            print('=' * 48)

    def train_and_test(self, dataset):
        train_ds = dataset
        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))

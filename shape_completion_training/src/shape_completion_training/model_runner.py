from shape_completion_training.model import utils

utils.set_gpu_with_lowest_memory()
import tensorflow as tf
from shape_completion_training.model import filepath_tools
from shape_completion_training.model.auto_encoder import AutoEncoder
from shape_completion_training.model.augmented_ae import Augmented_VAE
from shape_completion_training.model.voxelcnn import VoxelCNN
from shape_completion_training.model.vae import VAE, VAE_GAN
from shape_completion_training.model.conditional_vcnn import ConditionalVCNN
from shape_completion_training.model.ae_vcnn import AE_VCNN
import progressbar
import datetime
import time


def get_model_type(network_type):
    if network_type == 'VoxelCNN':
        return VoxelCNN
    elif network_type == 'AutoEncoder':
        return AutoEncoder
    elif network_type == 'VAE':
        return VAE
    elif network_type == 'VAE_GAN':
        return VAE_GAN
    elif network_type == 'Augmented_VAE':
        return Augmented_VAE
    elif network_type == 'Conditional_VCNN':
        return ConditionalVCNN
    elif network_type == 'AE_VCNN':
        return AE_VCNN
    else:
        raise Exception('Unknown Model Type')


class ModelRunner:
    def __init__(self, model, training, group_name=None, trial_path=None, params=None, write_summary=True):
        self.model = model
        self.side_length = 64
        self.num_voxels = self.side_length ** 3
        self.training = training

        self.trial_path, self.params = filepath_tools.create_or_load_trial(group_name=group_name,
                                                                           params=params,
                                                                           trial_path=trial_path,
                                                                           write_summary=write_summary)
        self.group_name = self.trial_path.parts[-2]

        self.batch_size = 16
        if not self.training:
            self.batch_size = 1

        self.train_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/train").as_posix())
        self.test_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/test").as_posix())

        self.num_batches = None

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        epoch=tf.Variable(0),
                                        train_time=tf.Variable(0.0),
                                        model=self.model)
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
        elem = next(iter(dataset.take(self.batch_size).batch(self.batch_size)))
        tf.summary.trace_on(graph=True, profiler=False)
        self.model(elem)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.ckpt.step.numpy())

        model_image_path = self.trial_path / 'network.png'
        tf.keras.utils.plot_model(self.model, model_image_path.as_posix(), show_shapes=True)

    def write_summary(self, summary_dict):
        with self.train_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())

    def train_batch(self, dataset):
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

                train_outputs, all_metrics = self.model.train_step(batch)
                time_str = str(datetime.timedelta(seconds=int(self.ckpt.train_time.numpy())))
                bar.update(self.num_batches, Loss=all_metrics['loss'].numpy().squeeze(), TrainTime=time_str)
                self.write_summary(all_metrics)
                self.ckpt.train_time.assign_add(time.time() - t0)
                t0 = time.time()

        save_path = self.manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        print("loss {:1.3f}".format(all_metrics['loss'].numpy()))

    def train(self, dataset, num_epochs):
        self.build_model(dataset)
        self.count_params()
        batched_ds = dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        while self.ckpt.epoch < num_epochs:
            self.ckpt.epoch.assign_add(1)
            print('')
            msg_fmt = '== Epoch {}/{} ========================= {} ===================='
            print(msg_fmt.format(self.ckpt.epoch.numpy(), num_epochs, self.group_name))
            self.train_batch(batched_ds)
            print('=' * 48)

    def train_and_test(self, dataset):
        train_ds = dataset
        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))

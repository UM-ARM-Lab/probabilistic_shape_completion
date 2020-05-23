from colorama import Style, Fore

from shape_completion_training.model import utils
from shape_completion_training.model.utils import reduce_mean_dict, sequence_of_dicts_to_dict_of_sequences

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
    def __init__(self,
                 model,
                 training,
                 group_name=None,
                 trial_path=None,
                 params=None,
                 trials_directory=None,
                 write_summary=True):
        self.model = model
        self.side_length = 64
        self.num_voxels = self.side_length ** 3
        self.training = training

        self.trial_path, self.params = filepath_tools.create_or_load_trial(group_name=group_name,
                                                                           params=params,
                                                                           trial_path=trial_path,
                                                                           trials_directory=trials_directory,
                                                                           write_summary=write_summary)
        self.group_name = self.trial_path.parts[-2]

        self.train_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/train").as_posix())
        self.val_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/val").as_posix())

        self.num_train_batches = None
        self.num_val_batches = None

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
        elem = next(iter(dataset))
        tf.summary.trace_on(graph=True, profiler=False)
        self.model(elem, training=True)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.ckpt.step.numpy())

        model_image_path = self.trial_path / 'network.png'
        tf.keras.utils.plot_model(self.model, model_image_path.as_posix(), show_shapes=True)

    def write_train_summary(self, summary_dict):
        with self.train_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())

    def write_val_summary(self, summary_dict):
        with self.val_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.ckpt.step.numpy())

    def train_epoch(self, train_dataset, val_dataset):
        if self.num_train_batches is not None:
            max_size = str(self.num_train_batches)
        else:
            max_size = '???'

        widgets = [
            '  ', progressbar.Counter(), '/', max_size,
            ' ', progressbar.Variable("Loss"), ' ',
            progressbar.Bar(),
            ' [', progressbar.Variable("TrainTime"), '] ',
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_train_batches) as bar:
            self.num_train_batches = 0
            t0 = time.time()
            for train_batch in train_dataset:
                self.num_train_batches += 1
                self.ckpt.step.assign_add(1)

                _, train_batch_metrics = self.model.train_step(train_batch)
                time_str = str(datetime.timedelta(seconds=int(self.ckpt.train_time.numpy())))
                bar.update(self.num_train_batches, Loss=train_batch_metrics['loss'].numpy().squeeze(), TrainTime=time_str)
                self.write_train_summary(train_batch_metrics)
                self.ckpt.train_time.assign_add(time.time() - t0)
                t0 = time.time()

        save_path = self.manager.save()
        print(Fore.CYAN + "Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path) + Fore.RESET)
        print("train loss {:1.3f}".format(train_batch_metrics['loss'].numpy()))

    def val_epoch(self, val_dataset):
        if self.num_val_batches is not None:
            max_size = str(self.num_val_batches)
        else:
            max_size = '???'

        widgets = [
            '  ', progressbar.Counter(), '/', max_size,
            progressbar.Bar(),
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_val_batches) as bar:
            self.num_val_batches = 0
            val_metrics = []
            for val_batch in val_dataset:
                self.num_val_batches += 1
                _, val_batch_metrics = self.model.val_step(val_batch)
                val_metrics.append(val_batch_metrics)
                bar.update(self.num_val_batches)

            val_metrics = sequence_of_dicts_to_dict_of_sequences(val_metrics)
            mean_val_metrics = reduce_mean_dict(val_metrics)
            self.write_val_summary(mean_val_metrics)
        print(Style.BRIGHT + "val loss {:1.3f}".format(mean_val_metrics['loss'].numpy()) + Style.NORMAL)

    def train(self, train_dataset, val_dataset, num_epochs, seed):
        self.build_model(train_dataset)
        self.count_params()

        try:
            while self.ckpt.epoch < num_epochs:
                # Training
                self.ckpt.epoch.assign_add(1)
                print('')
                msg_fmt = Fore.GREEN + Style.BRIGHT + 'Epoch {:3d}/{}, Group Name {}' + Style.RESET_ALL
                print(msg_fmt.format(self.ckpt.epoch.numpy(), num_epochs, self.group_name))
                self.train_epoch(train_dataset, val_dataset)

                # Validation at end of epoch
                self.val_epoch(val_dataset)
        except KeyboardInterrupt:
            print(Fore.YELLOW + "Interrupted." + Fore.RESET)

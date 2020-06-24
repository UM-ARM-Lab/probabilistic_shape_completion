import tensorflow as tf
from shape_completion_training.model.mykerasmodel import MyKerasModel
import tensorflow.keras.layers as tfl

from shape_completion_training.model.utils import stack_known, log_normal_pdf


def compute_vae_loss(z, mean, logvar, sample_logit, labels):
    # mean, logvar = model.encode(x)
    # z = model.reparameterize(mean, logvar)
    # x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logit, labels=labels)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_box_loss(gt_latent_box, inferred_latent_box_mean, inferred_latent_box_logvar):
    losses = -log_normal_pdf(gt_latent_box, inferred_latent_box_mean, inferred_latent_box_logvar)
    return tf.reduce_mean(losses)


class NormalizingAE(MyKerasModel):
    def __init__(self, hparams, batch_size, *args, **kwargs):
        super(NormalizingAE, self).__init__(hparams, batch_size, *args, **kwargs)
        self.flow = None
        self.encoder = make_encoder(inp_shape=[64, 64, 64, 2], params=hparams)
        self.generator = make_generator(params=hparams)
        self.box_latent_size = 24

    def call(self, dataset_element, training=False, **kwargs):
        known = stack_known(dataset_element)
        mean, logvar = self.encode(known)
        sampled_features = self.sample_latent(mean, logvar)
        predicted_occ=self.decode(sampled_features, apply_sigmoid=True)
        output = {'predicted_occ': predicted_occ, 'predicted_free': 1 - predicted_occ,
                  'latent_mean': mean, 'latent_logvar': logvar, 'sampled_latent': sampled_features}
        return output

    def sample_latent(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        features = eps * tf.exp(logvar * 0.5) + mean
        return features

    def split_box(self, inp):
        features, box = tf.split(inp, num_or_size_splits=[self.hparams['num_latent_layers'] - self.box_latent_size,
                                                          self.box_latent_size], axis=1)
        return features, box

    def replace_true_box(self, z, true_box):
        f, _ = self.split_box(z)
        return tf.concat([f, true_box], axis=1)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        logits = self.generator(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def predict(self, elem):
        return self(next(elem.__iter__()))

    # def compute_loss(self, gt_latent, train_outputs):
    #     losses = -log_normal_pdf(gt_latent, train_outputs['mean'], train_outputs['logvar'])
    #     loss = tf.reduce_mean(losses)
    #     return {"loss": loss}

    @tf.function
    def train_step(self, train_element):
        bb = tf.keras.layers.Flatten()(tf.cast(train_element['bounding_box'], tf.float32))
        gt_latent_box = self.flow.bijector.inverse(bb)
        gt_latent_box = tf.stop_gradient(gt_latent_box)
        # train_element['gt_latent_box'] = gt_latent_box
        with tf.GradientTape() as tape:
            # train_outputs = self.call(train_element, training=True)
            # train_losses = self.compute_loss(gt_latent_box, train_outputs)
            known = stack_known(train_element)
            mean, logvar = self.encode(known)
            sampled_latent = self.sample_latent(mean, logvar)
            corrected_latent = self.replace_true_box(sampled_latent, gt_latent_box)
            logits = self.decode(corrected_latent)

            mean_f, mean_box = self.split_box(mean)
            logvar_f, logvar_box = self.split_box(logvar)
            sampled_f, _ = self.split_box(sampled_latent)

            vae_loss = compute_vae_loss(sampled_f, mean_f, logvar_f, logits, train_element['gt_occ'])
            box_loss = compute_box_loss(gt_latent_box, mean_box, logvar_box)
            train_losses = {"loss/vae_loss": vae_loss, "loss/box_loss": box_loss,
                            "loss": vae_loss + box_loss}
            train_outputs = None

        gradient_metrics = self.apply_gradients(tape, train_element, train_outputs, train_losses)
        other_metrics = self.calculate_metrics(train_element, train_outputs)

        metrics = {}
        metrics.update(train_losses)
        metrics.update(gradient_metrics)
        metrics.update(other_metrics)

        return train_outputs, metrics


def make_encoder(inp_shape, params):
    """Basic VAE encoder"""
    n_features = params['num_latent_layers']

    return tf.keras.Sequential(
        [
            tfl.InputLayer(input_shape=inp_shape),

            tfl.Conv3D(64, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Conv3D(128, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Conv3D(256, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Conv3D(512, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Flatten(),
            tfl.Dense(n_features * 2)
        ]
    )


def make_generator(params):
    """Basic VAE decoder"""
    n_features = params['num_latent_layers']
    return tf.keras.Sequential(
        [
            tfl.InputLayer(input_shape=(n_features,)),
            tfl.Dense(4 * 4 * 4 * 512),
            tfl.Activation(tf.nn.relu),
            tfl.Reshape(target_shape=(4, 4, 4, 512)),

            tfl.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(1, (2, 2, 2), strides=(1, 1, 1), padding="same"),
        ]
    )

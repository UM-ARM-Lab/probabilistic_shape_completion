from timeit import timeit

import tensorflow as tf

optimizer = tf.keras.optimizers.Adam()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
])

element = tf.constant([[1]], dtype=tf.float32)


def compute_loss(dataset_element, outputs):
    return tf.keras.losses.MSE(dataset_element, outputs)


# This will work
@tf.function
def inline():
    with tf.GradientTape() as tape:
        outputs = model(element, training=True)
        loss = compute_loss(element, outputs)

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


def inline_no_tf_function():
    with tf.GradientTape() as tape:
        outputs = model(element, training=True)
        loss = compute_loss(element, outputs)

    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


# loss from outer scope also works
@tf.function
def outer_scope():
    with tf.GradientTape() as tape:
        outputs = model(element, training=True)
        loss = compute_loss(element, outputs)

    @tf.function
    def apply_gradients(tape):
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    apply_gradients(tape)


# loss passing into function does not work
def apply_gradients_no_tf_function(tape, loss):
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))


@tf.function
def passing_loss_no_tffunction():
    with tf.GradientTape() as tape:
        outputs = model(element, training=True)
        loss = compute_loss(element, outputs)

    apply_gradients_no_tf_function(tape, loss)


@tf.function
def passing_loss():
    with tf.GradientTape() as tape:
        outputs = model(element, training=True)
        loss = compute_loss(element, outputs)

    @tf.function
    def apply_gradients(tape, loss):
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    apply_gradients(tape, loss)


inline()
outer_scope()
passing_loss_no_tffunction()
# passing_loss()

from time import perf_counter

t0 = perf_counter()
for i in range(1000):
    passing_loss_no_tffunction()
print('{:.4f}'.format(perf_counter() - t0))

t0 = perf_counter()
for i in range(1000):
    inline()
print('{:.4f}'.format(perf_counter() - t0))

t0 = perf_counter()
for i in range(1000):
    inline_no_tf_function()
print('{:.4f}'.format(perf_counter() - t0))

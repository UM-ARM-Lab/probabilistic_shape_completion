import tensorflow as tf


def simulate_omniscient_input(gt):
    """
    Given a single ground truth mask occupied list,
    return ground truth occupied and free,
    as well as simulate the known occupied and free
    """
    return gt, 1.0 - gt


def simulate_partial_completion(dataset):
    def _add_partial_gt(elem):
        partial_occ, partial_free = simulate_first_random_input(elem['gt_occ'])
        elem['known_occ'] = tf.clip_by_value(elem['known_occ'] + partial_occ, 0.0, 1.0)
        elem['known_free'] = tf.clip_by_value(elem['known_free'] + partial_free, 0.0, 1.0)
        return elem

    return dataset.map(_add_partial_gt)


def simulate_random_partial_completion(dataset):
    def _add_partial_gt(elem):
        partial_occ, partial_free = simulate_random_partial_completion_input(elem['gt_occ'])
        elem['known_occ'] = tf.clip_by_value(elem['known_occ'] + partial_occ, 0.0, 1.0)
        elem['known_free'] = tf.clip_by_value(elem['known_free'] + partial_free, 0.0, 1.0)
        return elem

    return dataset.map(_add_partial_gt)


def simulate_condition_occ(dataset, turn_on_prob=0.0, turn_off_prob=0.0):
    def _add_conditional(elem):
        x = elem['gt_occ']
        x = x + tf.cast(tf.random.uniform(x.shape) < turn_on_prob, tf.float32)
        x = x - tf.cast(tf.random.uniform(x.shape) < turn_off_prob, tf.float32)
        x = tf.clip_by_value(x, 0.0, 1.0)
        elem['conditioned_occ'] = x
        return elem

    return dataset.map(_add_conditional)


def simulate_first_n_input(gt, n):
    gt_occ = gt
    gt_free = 1.0 - gt
    mask = tf.concat([tf.ones(n), tf.zeros(tf.size(gt) - n)], axis=0)
    shape = gt.shape
    gt_occ_masked = tf.reshape(tf.reshape(gt_occ, [-1]) * mask, shape)
    gt_free_masked = tf.reshape(tf.reshape(gt_free, [-1]) * mask, shape)
    return gt_occ_masked, gt_free_masked


def simulate_first_random_input(gt):
    n = tf.random.uniform(shape=[1], minval=0, maxval=tf.size(gt), dtype=tf.int32)
    return simulate_first_n_input(gt, n)


def simulate_random_partial_completion_input(gt):
    gt_occ = gt
    gt_free = 1.0 - gt

    # mask_n = tf.random.uniform(shape=[1], minval=0, maxval=tf.size(gt), dtype=tf.int32)
    mask_n = tf.random.uniform(shape=[1])
    mask_n = tf.pow(mask_n, 8.0)
    mask_n = tf.cast(mask_n * tf.cast(tf.size(gt), tf.float32), tf.int32)
    mask = tf.concat([tf.ones(mask_n), tf.zeros(tf.size(gt) - mask_n)], axis=0)

    mask = tf.random.shuffle(mask)

    shape = gt.shape

    gt_occ_masked = tf.reshape(tf.reshape(gt_occ, [-1]) * mask, shape)
    gt_free_masked = tf.reshape(tf.reshape(gt_free, [-1]) * mask, shape)
    return gt_occ_masked, gt_free_masked
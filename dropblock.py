import tensorflow as tf

def dropblock(x, keep_prob, block_size):
    _,w,h,c = x.shape.as_list()
    gamma = (1. - keep_prob) * (w * h) / (block_size ** 2) / ((w - block_size + 1) * (h - block_size + 1))
    sampling_mask_shape = tf.stack([1, h - block_size + 1, w - block_size + 1, c])
    noise_dist = tf.distributions.Bernoulli(probs=gamma)
    mask = noise_dist.sample(sampling_mask_shape)

    br = (block_size - 1) // 2
    tl = (block_size - 1) - br
    pad_shape = [[0, 0], [tl, br], [tl, br], [0, 0]]
    mask = tf.pad(mask, pad_shape)
    mask = tf.nn.max_pool(mask, [1, block_size, block_size, 1], [1, 1, 1, 1], 'SAME')
    mask = tf.cast(1 - mask,tf.float32)
    return tf.multiply(x,mask)

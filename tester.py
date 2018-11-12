from dropblock import dropblock
import tensorflow as tf
import numpy as np

[n,w,h,c] = [5,5,5,3]
item_num = n*w*h*c
input = np.arange(item_num).reshape([n,w,h,c])
with tf.Session() as sess:
    ph = tf.placeholder(shape=[None,w,h,c],dtype=tf.float32)
    model = dropblock(ph,0.1,3)
    out = sess.run(model,feed_dict={ph:input})

print(out.shape)
print(out[0,:,:,0])
print(np.sum(out==0,axis=None)/(item_num))
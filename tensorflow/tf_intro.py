#http://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf

import tensorflow as tf

tf.InteractiveSession()

a = tf.zeros((2,2))
#  <tf.Tensor 'zeros:0' shape=(2, 2) dtype=float32>

b = tf.ones([2,2])
# <tf.Tensor 'ones:0' shape=(2, 2) dtype=float32>

c = tf.reduce_sum(b, reduction_indices=1).eval()
# array([ 2.,  2.], dtype=float32)

a.get_shape()
#TensorShape([Dimension(2), Dimension(2)])

tf.reshape(a, (1,4)).eval()
#array([[ 0.,  0.,  0.,  0.]], dtype=float32)

W1  = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name = "weights")
with tf.Session() as sess:
    print sess.run(W1)
    tf.initialize_all_variables().run()
    print W2.eval()
    
state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
        
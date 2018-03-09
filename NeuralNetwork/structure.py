import numpy as np
import tensorflow as tf

from NeuralNetwork.Datas_second import load_datas


def initialize_parameters(layer_dim):
    parameters = {}
    L = len(layer_dim)

    for l in range(1, L):
        parameters["W" + str(l)] = tf.Variable(tf.random_normal([layer_dim[l], layer_dim[l - 1]]) * 0.01)
        parameters["b" + str(l)] = tf.Variable(tf.zeros([layer_dim[l], 1]))

    return parameters


xt = tf.placeholder(tf.float32,name="xt")
yt = tf.placeholder(tf.float32,name="yt")

layer_dim = [21, 64, 64, 64, 33]
parameters = initialize_parameters(layer_dim)
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
W3 = parameters["W3"]
b3 = parameters["b3"]
W4 = parameters["W4"]
b4 = parameters["b4"]
# 第一层隐藏层
h1_Z = tf.matmul(W1, xt) + b1
h1_A = tf.nn.relu(h1_Z)

# 第二层隐藏层
h2_Z = tf.matmul(W2, h1_A) + b2
h2_A = tf.nn.relu(h2_Z)

# 第三层隐藏层
h3_Z = tf.matmul(W3, h2_A) + b3
h3_A = tf.nn.relu(h3_Z)

# 输出层
h4_Z = tf.matmul(W4, h3_A) + b4
prediction = tf.nn.sigmoid(h4_Z)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=yt))
train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss=cross_entropy)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
X, Y = load_datas()
for i in range(160):
    batch_xt = np.array(X[:, i:i + 10])
    batch_yt = np.array(Y[:, i:i + 10])
    sess.run(train_step, feed_dict={xt: batch_xt, yt: batch_yt})
    print(sess.run(cross_entropy))

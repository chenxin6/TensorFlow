import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 这句话的意思是如果你电脑本地没有这个数据包则其会去网上下载相应的数据包
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
# 定义计算准确性的功能
def compute_accuracy(v_xs, v_ys):
    global prediction
    # 基于概率的预测值
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    # 对比预测值和真实数据的差别
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs})
    # result是一个百分比，该数值越高就越准确
    return result

# 784=28*28，表示每个图片有784个像素点作为输入
xs = tf.placeholder(tf.float32, [None, 784])
# 表示有10个输出来分类，这个10和上面的784都是Google规定好的
ys = tf.placeholder(tf.float32, [None, 10])

# 这个激励函数比较适合做分类学习
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
# 这个就是loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    # 从下载好的database提取出100个数据来进行学习
    # 相比于运用全部的数据来进行学习，这个方法会快些
    # 这就是所谓的随机梯度下降（SGD）
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    if i % 50 == 0:
        # 使用计算准确性的功能
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
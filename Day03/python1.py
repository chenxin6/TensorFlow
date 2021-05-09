import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 初值最好是略大于0的值
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# x是输入，strides是步长
def conv2d(x, W):
    # 第一个参数和第四个参数必须要是1
    # 第二个1是水平方向的步长（跨步），第三个1是竖直方向的步长（跨步）
    # padding有两种SAME和VALID
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling的处理是防止跨步太大导致图片信息的丢失
# 即想要全部的图片信息，又想要压缩图片（降低宽和高）就可以用pooling
def max_pool_2x2(x):
    # ksize是池化窗口（即卷积核）的大小，是一个四位向量
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 784]) # 784 = 28 * 28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# 第一个-1表示把所有图片（xs）的纬度先不管，之后再加这些纬度信息，即不知道填什么数据，但是可以通过后面的数据计算得出
# 第四个1代表图片是黑白色的所以channel只有一个，如果是彩色（RGB）的话channel的数量就是3
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape) # [n_samples, 28, 28, 1]

# conv1 layer
# 前两个5表示patch为5 * 5
# 1表示 in size 为1，即image的厚度为1，即黑白色的channel数量为1
# 32表示 out size 为32，压缩图片后的图片厚度为32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28 * 28 * 32（28 * 28是因为padding='SAME'）
h_pool1 = max_pool_2x2(h_conv1) # output size 14 * 14 * 32（14 * 14是因为padding='SAME'且ksize中设置步长为2）

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64]) # 注意和conv1 layer的区别，输入输出的size变化了，即图片厚度变大了
b_conv2 = bias_variable([64]) # 注意区别
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14 * 14 * 64
h_pool2 = max_pool_2x2(h_conv2) # output size 7 * 7 * 64

# func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1= bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # [n_samples, 7, 7, 64] >> [n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer
# 10代表分类结果可能性的数量，详见 Day02/python3.py
W_fc2 = weight_variable([1024, 10])
b_fc2= bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
# 类似于之前代码中的loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()    
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:1})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

import tensorflow as tf
import numpy as np

# 添加神经层的函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义一个矩阵有in_size行out_size列
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 定义一个1行out_size列的零矩阵，加0.1是因为biases不推荐初始值为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 从-1～1分成300份
# newaxis等价于None，作用是增加维度
# 原本一维数组就变成了行为数组长度，列为1的二维数组了
# :和np.newaxis对调的话就是变成行为1，列为数组长度的二维数组
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 加上噪声
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# [None, 1]表示行不定列是1
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义l1隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
# reduce_sum是求和用于降维
# reduction_indices是处理维度，可以有None，[0,1]，[1,0]，[0]和[1]
# 前三个的输出都是一样的都是输出为一个数，而[0]是纵向压缩，[1]是横向压缩
# 这里由于prediction和ys是多行1列的结构所以用横向压缩的处理
# reduce_mean是求平均值
loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
# 构建学习效率为0.1的优化器，这个优化器的任务是减小loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))



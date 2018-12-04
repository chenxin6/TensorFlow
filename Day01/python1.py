# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# creat data
x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3

# creat tensorflow structure start
# 一维结构，初始范围-1.0到1.0
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y_out = Weights * x + biases

loss = tf.reduce_mean(tf.square(y_out - y))
# 构建优化器减少误差，0.5是学习效率（0～1）
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# creat tensorflow structure end

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y)
# 连续的输入，这样就不会让程序暂停在这里了
# 否则的话图像页面没有关闭后面的代码就无法运行
plt.ion()

plt.show()
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
        try:
            # 这里把线抹去是为了显示出动态的效果
            # 没有这行的话就会图像密密麻麻的有很多线
            ax.lines.remove(lines[0])
            pass
        except Exception:
            pass
        prediction_value = sess.run(y_out)
        lines = ax.plot(x, prediction_value, 'r', lw=5)
        plt.pause(0.5)
        # ax.lines.remove(lines[0])

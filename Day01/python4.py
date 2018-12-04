import tensorflow as tf

# 规定数据类型
# 如果用placeholder的话则在run的时候才赋值
# 在run的时候其数值是通过feed_dict传进去的
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 注意multiply和matmul的区别
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))

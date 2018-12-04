import tensorflow as tf

# 定义两个常量
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

# 矩阵相乘
product = tf.matmul(matrix1, matrix2)

# method 1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# method 1
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)




import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)

one = tf.constant(1)

# 把这个变量理解成是一种状态，有点像Vue中数据驱动视图的感觉
# state一变它就立马跟着变
new_value = tf.add(state, one)

# 把new_value赋值会state
update = tf.assign(state, new_value)

# 激活所有的变量！！！很重要
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(new_value))
        print(sess.run(state))

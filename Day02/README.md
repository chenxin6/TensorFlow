# TensorFlow
## python1.py
TensorBoard可视化好帮手（入门）
## python2.py
TensorBoard可视化好帮手（进阶）
DISTRIBUTIONS和HISTOGRAMS都是频率分布直方图
前者便于观察权重值随学习进度的变化趋势
后者会显示具体的数值，所展示的信息会更多
## python3.py（重点）
概括来讲前面的学习都是线性回归的问题
而这部分要讲的是分类学习(Classification)
这里还引入的监督学习的概念，即代码中的compute_accuracy
## python4.py（重点）
用dropout解决过拟合的问题
过拟合（overfitting）说白了就是机器学习模型过于自信（自负）
但如果数据量够大就不存在过拟合的情况，但是实际上往往不可能获得到很多的数据
解决一：增加数据量
解决二：利用正规化（正则化），神经网络中常用的正规化方法是dropout(忽略掉一些神经元和神经元连接)说白了就是如果某些神经元或连接的权重太大就把他暂时架空，使得输出结果跟这个神经元或连接无关
需要加库numpy，scipy，scikit-learn
这里我已经有了numpy，所以我就运行以下两句
```
python -m pip install scipy
python -m pip install scikit-learn
```

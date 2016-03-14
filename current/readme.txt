1.现在all cart里面和之前一样也是user cart，但user cart里的现在不再是购买的物品而是behavior即［物品，时间间隔］  
2.现在一共分为10种时间间隔， time_interval是一个a＊b的矩阵，a是时间间隔的种类，b是每种时间间隔向量的长度  
3.Tt是张量，相当于纸上公式里的Tm  
4.3.8加入了matrix.py 和basicrnn.py分别用于在新数据集上实现原始的rnn以及multi matrix形式的rnn。先在小数据集上实现好算法再转战大数据集。



3.14
	13:38 ：整理文件夹并改名为current，在matrix.py中排查掉了一些bug，切分出了两个小数据集（300用户，100000条记录，itemid已进行了重排列操作。），目前在小数据集上继续查找代码中存在的问题。
    18:08 : 在小数据集跑basicrnn作为准确率参照，结果保存在resultbasic。在小数据集上跑了将近100轮matrix，结果保存在result11.将matrix中的time_cart改为固定值，仅使用11个矩阵中的1个或2个，分别对应程序matrix1和matrix，结果也对应保存在两个文件中。


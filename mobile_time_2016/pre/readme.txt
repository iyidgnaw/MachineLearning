1.现在all cart里面和之前一样也是user cart，但user cart里的现在不再是购买的物品而是behavior即［物品，时间间隔］  
2.现在一共分为10种时间间隔， time_interval是一个a＊b的矩阵，a是时间间隔的种类，b是每种时间间隔向量的长度  
3.Tt是张量，相当于纸上公式里的Tm  
4.3.8加入了matrix.py 和basicrnn.py分别用于在新数据集上实现原始的rnn以及multi matrix形式的rnn。先在小数据集上实现好算法再转战大数据集。

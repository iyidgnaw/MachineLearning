# MachineLearning
# Introduction
Here are the works of Lizk and WangDiYi in CASIA.
# Including
1. Codes for embedding&RNN are in the folder "RNN_lizk".Dealing with the One-basket problem, we put the embedding method into the recurrent nerual network. Until now, these part of works are not being well. If you have any idea or questions, pls contact us.
2. Codes for the libsvm: (in the script "testing.py"and "10_19_1.py")
3. To be continued



# Update Note
10.21 加入了循环控制，读写控制，可以保存结果，完成了第一版可以连夜一直在服务器上运行的程序  
10.22 加入了时间显示。检测每一轮训练和测试的时间长度，将程序分成了两个版本，一个是带负采样另一个不带。将loss函数中的计算方式调整为较简便的公式实现。  
10.26 为防止loss中的正负样本由于数量差异产生误差，加入bias变量。加入avr变量监测每个篮子的正确率。  
10.27 bias改为动态调整。为方便检查，将loss函数改回原来的计算方式。  
10.29 将重心转移到训练过程，在训练集上，检测每一个篮子的训练之后对其下一个篮子中的物品的预测准确率，每10轮检测一次，结果不容乐观。（命中/总数：7400/210000）  
10.30 加入自适应学习速率，继续监测在训练集上的拟合程度，比昨日有所提升。（命中/总数：8500/210000）且命中率仍然在不断提升，加负采样的情况训练速度比较慢，所以可能需要利用周末时间训练，周一再看具体结果了。  
10.31 将learning_rate 设为 1e-3，并保持自适应学习速率。另外， 加入valueTrue计算实际的value值，valuePredict计算预测的value值，并对其进行比较，然而发现valueTrue > valuePredict却没有预测到真实的值。
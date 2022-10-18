### 参考文献
https://zhuanlan.zhihu.com/p/430141027

https://blog.51cto.com/u_15273875/5550547
### 数据下载
* [Kaggel](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data)
* [百度网盘](https://pan.baidu.com/s/1QG6nUcHx0QDRrFWw2bcTJg) 提取码：p96v

### 数据浏览
图像大小不一样
图像名称就是标签
./train
    12500张 猫的图像
    12500张 狗的图像
./test
    12500张 混合图像, 命名是数字
    
### 数据加载
* 数据集生成
数据集生成的总体思路是继承torch.utils.data.Dataset类，自己实现getitem和len这两个私有方法来完成对自己数据的读取操作。其中getitem这个函数的主要功能是根据样本的索引，返回索引对应的一张图片的图像数据X和对应的标签文件Y，X和Y组成一个训练样本，即返回一个训练样本。len函数的功能是直接返回数据集的样本个数。








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
数据集生成的总体思路是继承torch.utils.data.Dataset类,自己实现getitem和len这两个私有方法来完成对自己数据的读取操作。其中getitem这个函数的主要功能是根据样本的索引,返回索引对应的一张图片的图像数据X和对应的标签文件Y,X和Y组成一个训练样本,即返回一个训练样本。len函数的功能是直接返回数据集的样本个数。

* 数据集划分
将训练集按照一定的比例划分成训练集, 验证集, 是为了得到准确率
为什么需要划分验证集?
    https://blog.roboflow.com/train-test-split/
    The train, validation, and testing splits are built to combat overfitting.

    The validation set is a separate section of your dataset that you will use during training to get a sense of how well your model is doing on images that are not being used in training.

    validation set metrics may have influenced you during the creation of the model,

* 数据加载
制作好的数据集不能直接送入模型中训练,需要使用一个数据加载器(dataloader)来加载数据,使用torch.utils.data.DataLoader()来将制作好的数据集划分成一个一个的batch用来后面训练,验证,测试的时候向网络中输入数据。

加载过之后数据形状由3维的(C, H, W)变成4维的(batchsize, C, H, W),多出了的一维是batch_size.

### 搭建网络
* 自己搭建一个网络
由两个卷积层和三个全连接层组成

nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
卷积网络的参数是什么意思?
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    
    in_channels (int) – Number of channels in the input image
    out_channels (int) – Number of channels produced by the convolution, 就是我想让输出多少通道，就设置为多少?
    kernel_size (int or tuple) – Size of the convolving kernel
    stride (int or tuple, optional) – Stride of the convolution. Default: 1
    padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0
    padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
    groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
    bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

    dilation 扩张
        https://blog.csdn.net/qq_34243930/article/details/107231539

    stride 步长










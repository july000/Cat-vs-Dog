import os
from random import shuffle
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from read_yaml import parse_yaml
from tqdm import tqdm

# 1.create dataset
class MyDataSet(Dataset):
    def __init__(self, data_path:str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        self.path_list = os.listdir(data_path)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalizer(min=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, idx:int):
        '''
        img to tensor, label to tensor
        将图片的索引idx转化成图片的路径, 然后使用PIL.Image读取图片, 
        然后使用torchvision.transforms现将图片resize成同一个大小(原始图片大小不一致), 
        再转化为tensor(即归一化), 再标准化(normalization), 这样得到图像数据X了.
        因为训练集的每一张图片的名字有猫狗的标签, 故对图片名进行split然后转成0, 1编码(狗:1, 猫0)
        '''
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)
        if self.train_flag:
            if img_path.split('.')[0] == 'dog':
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0])
        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label
    
    def __len__(self):
        '''
        通过计算图片路径列表中图片路径的个数来得到.
        '''
        return len(self.path_list)

# 2. dataset split 
def dataset_split(full_ds, train_rate):
    '''
    为了得到准确率的结果,需要在有标签的训练数据集中划分出一部分数据作为验证集来得到准确率这个指标
    '''
    train_size = int(len(full_ds) * train_rate)
    validate_size = len(full_ds) - train_size
    train_ds, validate_ds = torch.utils.data.random_split(full_ds, [train_size, validate_size])
    return train_ds, validate_ds

# 3.data loader
def datalodaer(dataset, batch_size):
    '''
    我们制作好的数据集不能直接送入模型中训练,需要使用一个数据加载器(dataloader)来加载数据,
    使用torch.utils.data.DataLoader()来将制作好的数据集划分成一个一个的batch
    用来后面训练,验证,测试的时候向网络中输入数据。
    '''
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    return data_loader

def debug():
    yaml_path = './config.yaml'
    cfg = parse_yaml(yaml_path)

    train_path = cfg['train_path']
    test_path = cfg['test_path']
    batch_size = cfg['batch_size']

    train_ds = MyDataSet(train_path)
    new_train_ds, validation_ds = dataset_split(train_ds, 0.8)
    test_ds = MyDataSet(test_path, train=False)
    new_train_loader = datalodaer(new_train_ds, batch_size)
    validation_loader = datalodaer(validation_ds, batch_size)
    test_loader = datalodaer(test_ds, batch_size)

    # testing train data can iterate or not
    for i, item in enumerate(tqdm(test_ds)):
        print(item)
        break
    for idx in range(train_ds.__len__()):
        img, label = train_ds.__getitem__(idx)
        print(img)
        print(label)
        break
    
    # testing data shape
    for i, item in enumerate(new_train_loader):
        print(item[0].shape)
        print(item[0])
        print(item[0].size(0))
        print(item[0].size(1))
        print(item[0].size(2))
        print(item[0].size(3))
        break
        

if __name__ == '__main__':
    debug()
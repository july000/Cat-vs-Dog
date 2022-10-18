import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from read_yaml import parse_yaml
from tqdm import tqdm

# 1.create dataset
class MyDateSet(Dataset):
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

def debug():
    yaml_path = './config.yaml'
    cfg = parse_yaml(yaml_path)
    train_path = cfg['train_path']
    test_path = cfg['test_path']

    train_ds = MyDataSet(train_path)
    test_ds = MyDataSet(test_path, train=False)
    for i, item in enumerate(tqdm(test_ds)):
        print(item)
        break
    for idx in range(train_ds.__len__()):
        img, label = train_ds.__getitem__(idx)
        print(img)
        print(label)
        break
    
# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob
import cv2
import os
import pandas as pd
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import h5py # 通过h5py读写hdf5文件

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class CIFAR10Dataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, data, target):
        super(CIFAR10Dataset, self).__init__()
        self.data = data
        self.target = target

    def __getitem__(self, index): # 该函数涉及到enumerate的返回值
        batch_x = self.data[index]
        batch_y = self.target[index]
        return batch_x, batch_y

    def __len__(self):
        return self.data.size(0)


def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def get_raw_cifar10_data(loader):
    '''
    @loader:传入一个DataLoader，借此获得源数据
    @return:返回原矩阵数据（将x，3,32,32调整成为x，32,32,3）
    '''
    train_data = []
    train_target = []

    # 循环得到训练数据
    for batch_idx, (data, target) in enumerate(loader):
        train_data.append(data.numpy())
        train_target.append(target.numpy())

    train_data = np.asarray(train_data)
    train_target = np.asarray(train_target)
    train_data = train_data.reshape([-1, 3, 32, 32])
    train_target = np.reshape(train_target, [-1])

    return train_data, train_target


def get_handled_cifar10_train_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/train.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/train.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
        # print("^_^ data loaded successfully from train.h5")
    else:
        h5_store = h5py.File("data/train.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    # 生成dataset的包装类
    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                  shuffle=shuffle)


def get_handled_cifar10_test_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/test.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/test.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
        print("^_^ data loaded successfully from test.h5")

    else:
        h5_store = h5py.File("data/test.h5", 'w')

        # 加载数据集
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()


    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    # 生成dataset的包装类
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                      shuffle=shuffle)


def data_aug(raw_img_data):
    '''
    首先对图片进行亮度调整，然后对图片进行随机的投射变换处理，少的地方进行补0处理
    @raw_img_data :[x,h,w,channel] numpy.array
    @return : [x,h,w,channel] numpy.array

    '''
    # print("input shape :{}, and will iterate {} times".format(raw_img_data.shape, np.size(raw_img_data, 0)))
    res = []
    for i in range(np.size(raw_img_data, 0)):
        # show raw image :
        # show(raw_img_data[i,:,:,:].reshape(32,32,3))

        # 对亮度进行随机调整：借助transforms.ColorJitter类，这是一个可调用对象，可以让类的实例像函数一样被调用,使用这个类需要将图片转成PIL Image
        img_data = raw_img_data[i, :, :, :]
        # img_data = np.transpose(img_data, [2, 0, 1])
        # 从PIL IMAGE转np得到的是[channel,height,width] np转PIL Image 也需要保证这一格式

        # print(img_data.shape)
        # print(img_data)
        # img_data = img_data.astype(np.uint8)

        img_data = transforms.ToPILImage()(torch.from_numpy(img_data))  # numpy->PIL Image # 假设是（32，32,3）而非（3,32,32）
        # img_data.show() # show raw image
        rand_brightness = np.random.rand()
        # print(rand_brightness)
        img_data = transforms.ColorJitter(brightness=rand_brightness)(img_data)  # modify brightness

        # 对图片进行透视变换
        transforms_proba = np.random.uniform(0.3, 1)  # probability of being perspectively transformed
        img_data = transforms.RandomPerspective(p=transforms_proba, distortion_scale=0.5)(img_data)

        # img_data.show()# show image augmented
        img_data = transforms.ToTensor()(img_data).numpy()  # PIL Image -> Tensor->numpy

        # reshape here
        res.append(img_data)

    # show augmented image
    # show(img_data)

    # print("^_^ data augmented finished with shape: {}".format(np.asarray(res).shape))
    return np.asarray(res)


def _transform_vector(self, width, x_shift, y_shift, im_scale, rot_in_degrees):
    """
    If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
    then it maps the output point (x, y) to a transformed input point
    (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
    where k = c0 x + c1 y + 1.
    The transforms are inverted compared to the transform mapping input points to output points.
    """

    rot = float(rot_in_degrees) / 90. * (math.pi / 2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)],
         [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image.
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift / (2 * im_scale))
    b2 = y_origin_delta - (y_shift / (2 * im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
    # initrialize
    data = []

    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

def get_train_raw_data():
    '''
    :return: train_image ,  train_target  | tensor
    '''
    if os.path.exists("data/train.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/train.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
        # print("^_^ data loaded successfully from train.h5")
    else:
        h5_store = h5py.File("data/train.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    # 生成dataset的包装类
    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    return train_data,train_target

if __name__ == '__main__':
    data = datagenerator(data_dir='data/Train400')
    data,target = get_train_raw_data()
    print(data.shape)
    print(target.shape)

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       
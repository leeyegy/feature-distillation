from scipy import fftpack
from PIL import Image
import math
import numpy as np
import argparse
from data_generator import *
from torch.utils.data import  DataLoader
import os
import h5py
import torch
from torchvision import transforms

def load_quantization_table(component, qs=40):
    # Quantization Table for JPEG Standard: https://tools.ietf.org/html/rfc2435
    if component == 'lum':
        q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    elif component == 'chrom':
        q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])
    elif component == 'dnn':
        q = np.array([[ 0,  0,  0,  0,  0,  1,  1,  1],
                      [ 0,  0,  0,  0,  1,  1,  1,  1],
                      [ 0,  0,  0,  1,  1,  1,  1,  1],
                      [ 0,  0,  1,  1,  1,  1,  1,  1],
                      [ 0,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1],
                      [ 1,  1,  1,  1,  1,  1,  1,  1]])
        q = q * qs + np.ones_like(q)    
    return q

def make_table(component, factor, qs=40):
    factor = np.clip(factor, 1, 100)
    if factor < 50:
        q = 5000 / factor
    else:
        q = 200 - factor * 2
    qt = (load_quantization_table(component, qs) * q + 50) / 100
    qt = np.clip(qt, 1, 255)
    return qt

def quantize(block, component, factor=100):
    qt = make_table(component, factor)
    return (block / qt).round()

def dequantize(block, component, factor=100):
    qt = make_table(component, factor)
    return block * qt

def dct2d(block):
    dct_coeff = fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'),
                            axis=1, norm='ortho')
    return dct_coeff

def idct2d(dct_coeff):
    block = fftpack.idct(fftpack.idct(dct_coeff, axis=0, norm='ortho'),
                         axis=1, norm='ortho')
    return block

def encode(npmat, component, factor):
    rows, cols = npmat.shape[0], npmat.shape[1]
    blocks_count = rows // 8 * cols // 8
    quant_matrix_list = []
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            for k in range(3):
                block = npmat[i:i+8, j:j+8, k] - 128.
                dct_matrix = dct2d(block)
                if component == 'jpeg':
                    quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom', factor)
                else:
                    quant_matrix = quantize(dct_matrix, component, factor)
                quant_matrix_list.append(quant_matrix)
    return blocks_count, quant_matrix_list

def decode(blocks_count, quant_matrix_list, component, factor):
    block_side = 8
    image_side = int(math.sqrt(blocks_count)) * block_side
    blocks_per_line = image_side // block_side
    npmat = np.empty((image_side, image_side, 3))
    quant_matrix_index = 0
    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_side
        j = block_index % blocks_per_line * block_side
        for c in range(3):
            quant_matrix = quant_matrix_list[quant_matrix_index]
            quant_matrix_index += 1
            if component == 'jpeg':
                dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom', factor)
            else:
                dct_matrix = dequantize(quant_matrix, component, factor)
            block = idct2d(dct_matrix)
            npmat[i:i+8, j:j+8, c] = block + 128.
    npmat = np.clip(npmat.round(), 0, 255).astype('uint8')
    return npmat

def jpeg(npmat, component='jpeg', factor=50):
    cnt, coeff = encode(npmat, component, factor)
    npmat_decode = decode(cnt, coeff, component, factor)
    return npmat_decode
def feature_distillation(image,args,batch_idx,target):
    # h5 -> *255 -> feature distillation -> /255 -> h5
    '''
    :param image: np.array | [BATCH_SIZE,C,H,W] | [0,1]
    :return: np.array | [BATCH_SIZE,C,H,W] | [0,1]
    '''
    res = []
    for i in range(image.shape[0]):


        # first version
        tmp_data = np.transpose(np.array(image[i], dtype='float'),[1,2,0])
        tmp_data = np.array(tmp_data, dtype='float')*255
        image_uint8 = (tmp_data).astype('uint8')

        ycbcr = Image.fromarray(image_uint8, 'RGB').convert('YCbCr')

        npmat = np.array(ycbcr)
        npmat_jpeg = jpeg(npmat, component=args.component, factor=args.factor)
        image_obj = Image.fromarray(npmat_jpeg, 'YCbCr').convert('RGB')
        res.append(transforms.ToTensor()(image_obj).numpy())
    res = np.asarray(res)
    res = np.reshape(res,[-1,3,32,32])
    return res






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--image', type=str, default='fig/lena.png', help='image name')
    parser.add_argument('--component', type=str, default='jpeg',
                        help='dnn-oriented or jpeg standard')
    parser.add_argument('--factor', type=int, default=50, help='compression factor')
    parser.add_argument("--attack_method",default = "PGD",type=str,choices=["PGD","FGSM","STA","Momentum","none","CW","DeepFool"])
    parser.add_argument("--epsilon",type=float,default=8/255)

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

    #load network
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--num_classes', default=10, type=int)
    args = parser.parse_args()


    #load data
    test_file_dir = os.path.join("data","tiny-imagenet",args.attack_method,str(args.epsilon))
    if not os.path.exists(test_file_dir):
        os.makedirs(test_file_dir) # make new dirs iteratively

    test_file_path =  os.path.join(test_file_dir,"test_denoiser.h5")
    if os.path.exists(test_file_path):
        print("%s already exists! = =" % test_file_path)
        h5_store = h5py.File(test_file_path,"r")
        denoised_data=h5_store["data"][:]
        target_ = h5_store["target"][:]
        h5_store.close()
    else:
        h5_store = h5py.File(test_file_path,"w")
        # generate (denoised_data,target)
        '''
        h5 -> *255 -> feature distillation -> /255 -> h5
        '''
        denoised_data = []
        target_ = []

        if args.attack_method == "none":
            testLoader = get_handled_tiny_imagenet_test_loader(batch_size=50, num_workers=2, shuffle=False)
        else:
            testLoader = get_test_adv_loader(attack_method = args.attack_method,epsilon=args.epsilon,args=args)


        for batch_idx,(data,target) in enumerate(testLoader):
            denoised_data.append(feature_distillation(data.numpy(),args,batch_idx,target))
            target_.append(target.numpy())
        denoised_data = np.reshape(denoised_data,[-1,3,64,64])
        target_= np.reshape(target_,[-1])
        # print("denoised data shape:{},  target shape:{}".format(denoised_data.shape,target_.shape))
        h5_store.create_dataset('data',data=denoised_data)
        h5_store.create_dataset('target',data=target_)
        h5_store.close()
    denoised_data = torch.from_numpy(denoised_data)
    target_ = torch.from_numpy(target_)

    test_denoised_dataset=CIFAR10Dataset(denoised_data,target_)
    del denoised_data,target_
    test_denoised_loader =  DataLoader(dataset=test_denoised_dataset,drop_last=True,batch_size=50,shuffle=False)

    # load network
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    model = torch.load('./checkpoint/resnet50_epoch_22.pth')  # os.sep提供跨平台的分隔符
    model = model.cuda()


    # evaluate
    model.eval()
    for epoch in range(1):
        clncorrect_nodefence = 0
        for batch_idx, (denoised_data, target) in enumerate(test_denoised_loader):
            denoised_data, target = denoised_data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(denoised_data.float())
            pred = output.max(1, keepdim=True)[1]
            clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()  # item： to get the value of tensor
        print('\nTest set with feature-dis defence epoch:{}'
                  ' cln acc: {}/{} ({:.0f}%)\n'.format(epoch,
                    clncorrect_nodefence, len(test_denoised_loader.dataset),
                      100. * clncorrect_nodefence / len(test_denoised_loader.dataset)))
        with open('./logs/' + args.attack_method + '@' + str(args.epsilon) + '.txt', 'a+') as f:
            f.write('epoch: %d succ_num: %d succ_rate: %f attack_method: %s epsilon: %f\n' % (
            epoch, clncorrect_nodefence, 100. * clncorrect_nodefence / len(test_denoised_loader.dataset), args.attack_method, args.epsilon))


if __name__ == '__main__':
    main()    

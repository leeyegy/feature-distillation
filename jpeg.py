from scipy import fftpack
from PIL import Image
import math
import numpy as np
import argparse
from data_generator import CIFAR10Dataset,get_test_adv_loader
from torch.utils.data import  DataLoader

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

def feature_distillation(image):
    # h5 -> *255 -> feature distillation -> /255 -> h5
    '''
    :param image: np.array | [BATCH_SIZE,C,H,W] | [0,1]
    :return: np.array | [BATCH_SIZE,C,H,W] | [0,1]
    '''
    res = []
    for i in range(image.shape[0]):
        image_npmat = np.transpose(np.array(image[i], dtype='float'),[2,0,1]) * 255
        image_uint8 = (image_npmat).astype('uint8')
        ycbcr = Image.fromarray(image_uint8, 'RGB').convert('YCbCr')
        npmat = np.array(ycbcr)
        npmat_jpeg = jpeg(npmat, component=args.component, factor=args.factor)
        image_obj = Image.fromarray(npmat_jpeg, 'YCbCr').convert('RGB')
        image_obj = np.asarray(image_obj) / 255
        res.append(image_obj)
    res = np.asarray(res)
    print ("res.shape:{}".format(res.shape))
    res = np.reshape(res,[-1,3,32,32])
    return res





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='fig/lena.png', help='image name')
    parser.add_argument('--component', type=str, default='jpeg',
                        help='dnn-oriented or jpeg standard')
    parser.add_argument('--factor', type=int, default=50, help='compression factor')
    parser.add_argument("--attack_method",type=str,choices=["PGD","FGSM","STA","Momentum"])
    parser.add_argument("--epsilon",type=float,default=8/255)
    args = parser.parse_args()

    test_file_dir = os.path.join("data",args.attack_method,args.epsilon)
    if not os.path.exists(test_file_dir):
        os.mkdir(test_file_dir)
    test_file_path =  os.path.join("data",args.attack_method,args.epsilon,"test_denoiser.h5")
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
        testLoader = get_test_adv_loader(attack_method = args.attack_method,epsilon=args.epsilon)
        for batch_idx,(data,target) in enumearte(testLoader):
            denoised_data.append(feature_distillation(data))
            target_.append(target)
        denoised_data = np.reshape(denoised_data,[-1,3,32,32])
        target_ = np.reshape(target_,[-1])
        h5_stroe.create_dataset('data',data=denoised_data)
        h5_store.create_dataset('target',data=target_)
        h5_store.close()
    denoised_data = torch.from_numpy(denoised_data)
    target_ = torch.from_numpy(target_)

    test_denoised_dataset=CIFAR10Dataset(denoised_data,target_)
    del denoised_data,target_
    return DataLoader(dataset=test_denoised_dataset,drop_last=True,batch_size=50,shuffle=False)


if __name__ == '__main__':
    main()    

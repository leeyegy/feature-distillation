from BPDA import  BPDAattack
import  numpy as np
import torch
from jpeg import feature_distillation,getNetwork
import argparse
from data_generator import  get_handled_cifar10_test_loader
import os

def fd(data,args):
    '''
    :param data: tensor.cuda() | [N,C,H,W] | [0,1]
    :param args: parameter setting
    :return: tensor.cuda() | [N,C,H,W] | [0,1]
    '''
    return torch.from_numpy(feature_distillation(data.cpu().numpy(),args,None,None)).cuda()

def main(args):
    # load data
    testLoader = get_handled_cifar10_test_loader(batch_size=50, num_workers=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # load model
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    print(file_name)
    checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')  # os.sep提供跨平台的分隔符
    model = checkpoint['net']
    model = model.to(device)

    # define adversary
    adversary = BPDAattack(model, fd, device,
                                epsilon=args.epsilon,
                                learning_rate=0.01,
                                max_iterations=args.max_iterations,args=args)

    # model test
    model.eval()
    clncorrect_nodefence = 0
    for data,target in testLoader:
        data, target = data.cuda(), target.cuda()
        # attack
        adv_data = adversary.perturb(data,target)

        # defence
        denoised_data = fd(adv_data,args)
        with torch.no_grad():
            output = model(denoised_data.float())
        pred = output.max(1, keepdim=True)[1]
        pred = pred.double()
        target = target.double()
        clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()  # item： to get the value of tensor
    print('\nTest set with feature-dis defence against BPDA'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                clncorrect_nodefence, len(testLoader.dataset),
                  100. * clncorrect_nodefence / len(testLoader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--image', type=str, default='fig/lena.png', help='image name')
    parser.add_argument('--component', type=str, default='jpeg',
                        help='dnn-oriented or jpeg standard')
    parser.add_argument('--factor', type=int, default=50, help='compression factor')
    parser.add_argument("--attack_method",default = "PGD",type=str,choices=["PGD","FGSM","STA","Momentum","none","DeepFool","CW","BIM"])
    parser.add_argument("--epsilon",type=float,default=8/255)

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

    #load network
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--num_classes', default=10, type=int)


    # BPDA ATTACK
    parser.add_argument("--max_iterations",default=10,type =int)
    args = parser.parse_args()
    main(args)
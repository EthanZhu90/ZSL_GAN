import torch
from torch.autograd import Variable

import pickle
import os
import sys
from termcolor import cprint
from sklearn.metrics.pairwise import cosine_similarity

import scipy.io as sio
import numpy as np
import argparse

from dataset import get_testing_text_feature
from models import _netG, _param


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--splitmode', default='easy', type=str, help='the way to split train/test data: easy/hard')
opt = parser.parse_args()

print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.nSample = 60  # number of fake feature for each class
opt.Knn = 20      # knn: the value of K


opt.resume = 'out/CUB_EASY/Eu1_Rls0.001_RWz0.0001/D5_3000.tar'

"""Custom the data path """

txt_feat_path  = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
if opt.splitmode == 'easy':
    train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
    pfc_label_path = 'data/CUB2011/labels_test.pkl'
    pfc_feat_path_train = 'data/CUB2011/pfc_feat_train.mat'
    pfc_feat_path = 'data/CUB2011/pfc_feat_test.mat'
    test_cls_num = 50
else:
    train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
    pfc_label_path = 'data/CUB2011/labels_test_hard.pkl'
    pfc_feat_path_train = 'data/CUB2011/pfc_feat_train_hard.mat'
    pfc_feat_path = 'data/CUB2011/pfc_feat_test_hard.mat'
    test_cls_num = 40


def generate_fakefeat_test():
    param = _param()
    test_text_feature = get_testing_text_feature(txt_feat_path, train_test_split_dir)
    print test_text_feature.shape
    text_dim = test_text_feature.shape[1]

    netG = _netG(text_dim).cuda()
    print(netG)

    if os.path.isfile(opt.resume):
        cprint("=> loading checkpoint '{}'".format(opt.resume), 'red')
        checkpoint = torch.load(opt.resume)
        netG.load_state_dict(checkpoint['state_dict_G'])
        print(checkpoint['log'])
    else:
        sys.exit("=> no checkpoint found at '{}'".format(opt.resume))

    gen_feat = np.zeros([0, param.X_dim])
    for i in range(test_cls_num):
        text_feat = np.tile(test_text_feature[i].astype('float32'), (opt.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).cuda()
        z = Variable(torch.randn(opt.nSample, param.z_dim)).cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)
    pfc_feat_data_train = pfc_feat_data_train['pfc_feat'].astype(np.float32)

    pfc_feat_data = sio.loadmat(pfc_feat_path)
    pfc_feat_data = pfc_feat_data['pfc_feat'].astype(np.float32)

    cprint('Test feature load from: {}'.format(pfc_feat_path), 'red')

    # Normalized testing data by centers of training data
    mean = pfc_feat_data_train.mean()
    var = pfc_feat_data_train.var()
    pfc_feat_data = (pfc_feat_data - mean) / var

    with open(pfc_label_path, 'rb') as output:
        labels = pickle.load(output)

    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(pfc_feat_data, gen_feat)
    idx_mat = np.argsort(-1*sim, axis=1)
    label_mat = idx_mat[:, 0:opt.Knn] / opt.nSample
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]
    acc = 100.0 * (np.asarray(labels) == preds).sum() / float(len(labels))
    print("{}nn with Cosine: ".format(opt.Knn))
    print("Accuracy is {:.4}%".format(acc))


if __name__ == "__main__":
    generate_fakefeat_test()


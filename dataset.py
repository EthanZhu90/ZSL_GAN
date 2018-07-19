import numpy as np
import scipy.io as sio
from termcolor import cprint
import pickle
import sys

class LoadDataset(object):
    def __init__(self, opt):
        txt_feat_path = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test.mat'
            train_cls_num = 150
            test_cls_num = 50
        else:
            train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test_hard.mat'
            train_cls_num = 160
            test_cls_num = 40

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        cprint("pfc_feat_file: {} || {} ".format(pfc_feat_path_train, pfc_feat_path_test), 'red')

        self.train_cls_num = train_cls_num
        self.test_cls_num  = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test  = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test  = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        self.tr_cls_centroid = np.zeros([train_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir)
        self.text_dim = self.train_text_feature.shape[1]


class LoadDataset_NAB(object):
    def __init__(self, opt):
        txt_feat_path = 'data/NABird/NAB_Porter_13217D_TFIDF_new.mat'
        if opt.splitmode == 'easy':
            train_test_split_dir = 'data/NABird/train_test_split_NABird_easy.mat'
            pfc_label_path_train = 'data/NABird/labels_train.pkl'
            pfc_label_path_test = 'data/NABird/labels_test.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_easy.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_easy.mat'
            train_cls_num = 323
            test_cls_num = 81
        else:
            train_test_split_dir = 'data/NABird/train_test_split_NABird_hard.mat'
            pfc_label_path_train = 'data/NABird/labels_train_hard.pkl'
            pfc_label_path_test = 'data/NABird/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_hard.mat'
            train_cls_num = 323
            test_cls_num = 81

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)
        cprint("pfc_feat_file: {} || {} ".format(pfc_feat_path_train, pfc_feat_path_test), 'red')

        self.train_cls_num = train_cls_num
        self.test_cls_num  = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test  = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test  = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        self.tr_cls_centroid = np.zeros([train_cls_num, self.pfc_feat_data_train.shape[1]]).astype(np.float32)
        for i in range(train_cls_num):
            self.tr_cls_centroid[i] = np.mean(self.pfc_feat_data_train[self.labels_train == i], axis=0)

        self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir)
        self.text_dim = self.train_text_feature.shape[1]


class FeatDataLayer(object):
    def __init__(self, label, feat_data,  opt):
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batchsize >= len(self._label):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._opt.batchsize]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels':minibatch_label}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs


def get_text_feature(dir, train_test_split_dir):
    train_test_split = sio.loadmat(train_test_split_dir)
    # get training text feature
    train_cid = train_test_split['train_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    train_text_feature = text_feature[train_cid - 1]  # 0-based index
    # get testing text feature
    test_cid = train_test_split['test_cid'].squeeze()
    text_feature = sio.loadmat(dir)['PredicateMatrix']
    test_text_feature = text_feature[test_cid - 1]  # 0-based index
    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)
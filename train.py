import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init


from termcolor import cprint
from time import gmtime, strftime
import scipy.io as sio
import numpy as np
import argparse
import pickle
import os

from dataset import FeatDataLayer, get_training_text_feature
from models import _netD, _netG, _param


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--splitmode', default='easy', type=str, help='the way to split train/test data: easy/hard')

opt = parser.parse_args()
print(opt)

opt.splitmode = 'hard'

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


""" hyper-parameter  """
opt.GP_LAMBDA = 10                # Gradient penalty lambda hyperparameter
opt.CENT_LAMBDA   = 1
opt.REG_W_LAMBDA = 0.001
opt.REG_Wz_LAMBDA = 0.0001

opt.lr = 0.001
opt.batchsize = 1000
opt.resume = None  # model path to resume


"""Custom the data path """


txt_feat_path  = 'data/CUB2011/CUB_Porter_7551D_TFIDF_new.mat'
if opt.splitmode == 'easy':
    train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
    pfc_label_path = 'data/CUB2011/labels_train.pkl'
    pfc_feat_path = 'data/CUB2011/pfc_feat_train.mat'
    train_cls_num = 150

else:
    train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
    pfc_label_path = 'data/CUB2011/labels_train_hard.pkl'
    pfc_feat_path = 'data/CUB2011/pfc_feat_train_hard.mat'
    train_cls_num = 160

def train():
    param = _param()
    pfc_feat_data = sio.loadmat(pfc_feat_path)
    pfc_feat_data = pfc_feat_data['pfc_feat'].astype(np.float32)
    cprint("pfc_feat_file: {}".format(pfc_feat_path), 'red')

    # calculate the corresponding centroid.
    with open(pfc_label_path, 'rb') as output:
        labels = pickle.load(output)
    tr_cls_centroid = np.zeros([train_cls_num, pfc_feat_data.shape[1]]).astype(np.float32)
    for i in range(train_cls_num):
        tr_cls_centroid[i] = np.mean(pfc_feat_data[labels == i], axis=0)

    # Normalize feat_data to zero-centered
    mean = pfc_feat_data.mean()
    var  = pfc_feat_data.var()
    pfc_feat_data = (pfc_feat_data - mean)/var

    data_layer = FeatDataLayer(labels, pfc_feat_data, opt)

    train_text_feature = get_training_text_feature(txt_feat_path, train_test_split_dir)
    text_dim = train_text_feature.shape[1]

    netG = _netG(text_dim).cuda()
    netG.apply(weights_init)
    print(netG)
    netD = _netD(train_cls_num).cuda()
    netD.apply(weights_init)
    print(netD)

    exp_info = 'CUB_EASY' if opt.splitmode == 'easy' else 'CUB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(opt.CENT_LAMBDA , opt.REG_W_LAMBDA, opt.REG_Wz_LAMBDA)

    out_dir  = 'out/{:s}'.format(exp_info)
    out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir  = out_subdir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(tr_cls_centroid.astype('float32'))).cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    for it in range(start_step, 3000+1):
        """ Discriminator """
        if it in {2000, 2500}:
            opt.lr = opt.lr/5.0
            optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
            optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))
            cprint("Learning rate:  {}".format(opt.lr), 'red')

        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']             # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([ train_text_feature[i,:] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()
            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)
            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).cuda()

            X = Variable(torch.from_numpy(feat_data)).cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).cuda()
            z = Variable(torch.randn(opt.batchsize, param.z_dim)).cuda()

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _,      C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true))/2

            GC_loss = -G_loss + C_loss

            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for i in range(train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        Euclidean_loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                Euclidean_loss *= 1.0/train_cls_num * opt.CENT_LAMBDA

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(opt.REG_W_LAMBDA)

            # ||W_z||21 regularization, make W_z sparse
            reg_Wz_loss = Variable(torch.Tensor([0.0])).cuda()
            if opt.REG_Wz_LAMBDA != 0:
                Wz = netG.rdc_text.weight
                reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(opt.REG_Wz_LAMBDA)

            all_loss = GC_loss + Euclidean_loss + reg_loss + reg_Wz_loss
            all_loss.backward()
            optimizerG.step()
            reset_grad(nets)

        if it % 50 == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(y_true.data.size()[0])

            log_text = 'Iter-{}; Was_D: {:.4}; Euc_ls: {:.4}; reg_ls: {:.4}; Wz_ls: {:.4}; G_loss: {:.4}; D_loss_real: {:.4};' \
                       ' D_loss_fake: {:.4}; rl: {:.4}%; fk: {:.4}%'\
                        .format(it, Wasserstein_D.data[0],  Euclidean_loss.data[0], reg_loss.data[0],reg_Wz_loss.data[0],
                                G_loss.data[0], D_loss_real.data[0], D_loss_fake.data[0], acc_real * 100, acc_fake * 100)
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text+'\n')

        if it % 500 == 0 and it != 0:
            torch.save({
                    'it': it + 1,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'log': log_text,
                },  out_subdir + '/D{:d}_{:d}.tar'.format(5, it))
            cprint('Save model to '+ out_subdir + '/D{:d}_{:d}.tar'.format(5, it), 'red')


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()


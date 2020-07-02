"""
Trainer for training and testing the networks
"""
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
import os
import numpy as np
import datetime
import json
import argparse
import glob
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from read_data import ChestXrayDataSet, ChestXrayDataSetTest

import seaborn as sn
import pandas as pd
from scipy import interp
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
from covidaid import CovidAID
from tqdm import tqdm

import math
import random
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
from collections.abc import Sequence, Iterable
import warnings
import version_support_functional as F


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)

class Trainer:
    def __init__ (self, local_rank=None, checkpoint=None, combine_pneumonia=False):
        """
        Trainer for the CovidAID
        """
        self.distributed = True if local_rank is not None else False
        print ("Distributed training %s" % ('ON' if self.distributed else 'OFF'))
        if self.distributed:
            raise NotImplementedError("Currently distributed training not supported")
            self.device = torch.cuda.device('cuda', local_rank)
        else:
            self.device = torch.cuda.device('cuda')

        # Using 2 classes for pneumonia vs 1 class
        self.combine_pneumonia = combine_pneumonia

        # self.net = CovidAID().to(self.device)
        self.net = CovidAID(combine_pneumonia).cuda()
        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)
        
        # load model
        if checkpoint is not None:
            self.load_model(checkpoint)

    def train(self, TRAIN_IMAGE_LIST, VAL_IMAGE_LIST, NUM_EPOCHS=10, LR=0.001, BATCH_SIZE=64,
                start_epoch=0, logging=True, save_path=None, freeze_feature_layers=True, gamma=2.0):
        """
        Train the CovidAID
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        train_dataset = ChestXrayDataSet(image_list_file=TRAIN_IMAGE_LIST,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            RandomAffine(30, scale=(0.8,1.2), shear=[-15,15,-15,15]),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]),
                                        combine_pneumonia=self.combine_pneumonia)
        if self.distributed:
            sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=8, pin_memory=True,
                                    sampler=sampler)
        else:
            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=8, pin_memory=True)

        val_dataset = ChestXrayDataSet(image_list_file=VAL_IMAGE_LIST,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]),
                                        combine_pneumonia=self.combine_pneumonia)
        if self.distributed:
            sampler = DistributedSampler(val_dataset)
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=8, pin_memory=True,
                                    sampler=sampler)
        else:
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=8, pin_memory=True)

        # Freeze heads and create optimizer
        if freeze_feature_layers:
            print ("Freezing feature layers")
            for param in self.net.densenet121.features.parameters():
                param.requires_grad = False

        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()),
        #                 lr=LR, momentum=0.9)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=LR)


        for epoch in range(start_epoch, NUM_EPOCHS):
            # switch to train mode
            self.net.train()
            tot_loss = 0.0
            for i, (inputs, target) in tqdm(enumerate(train_loader), total=len(train_dataset)/BATCH_SIZE):
                # inputs = inputs.to(self.device)
                # target = target.to(self.device)
                inputs = inputs.cuda()
                target = target.cuda()

                # Shape of input == [BATCH_SIZE, NUM_CROPS=19, CHANNELS=3, HEIGHT=224, WIDTH=244]
                bs, n_crops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                inputs = torch.autograd.Variable(inputs.view(-1, c, h, w))
                target = torch.autograd.Variable(target)
                preds = self.net(inputs).view(bs, n_crops, -1).mean(dim=1)

                # loss = torch.sum(torch.abs(preds - target) ** 2)    
                loss = train_dataset.loss(preds, target, gamma)  
                # exit()          
                tot_loss += float(loss.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss /= len(train_dataset)

            # Clear cache
            torch.cuda.empty_cache()
            
            # Running on validation set
            self.net.eval()
            val_loss = 0.0
            for i, (inputs, target) in tqdm(enumerate(val_loader), total=len(val_dataset)/BATCH_SIZE):
                # inputs = inputs.to(self.device)
                # target = target.to(self.device)
                inputs = inputs.cuda()
                target = target.cuda()

                # Shape of input == [BATCH_SIZE, NUM_CROPS=19, CHANNELS=3, HEIGHT=224, WIDTH=244]
                bs, n_crops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                inputs = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)
                target = torch.autograd.Variable(target, volatile=True)

                preds = self.net(inputs).view(bs, n_crops, -1).mean(1)
                # loss = torch.sum(torch.abs(preds - target) ** 2)
                loss = val_dataset.loss(preds, target, gamma) 
                
                val_loss += float(loss.data)

            val_loss /= len(val_dataset)

            # Clear cache
            torch.cuda.empty_cache()

            # logging statistics
            timestamp = str(datetime.datetime.now()).split('.')[0]
            log = json.dumps({
                'timestamp': timestamp,
                'epoch': epoch+1,
                'train_loss': float('%.5f' % tot_loss),
                'val_loss': float('%.5f' % val_loss),
                'lr': float('%.6f' % LR)
            })
            if logging:
                print (log)

            log_file = os.path.join(save_path, 'train.log')
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write("{}\n".format(log))

            model_path = os.path.join(save_path, 'epoch_%d.pth'%(epoch+1))
            self.save_model(model_path)
            
        print ('Finished Training')

    def predict(self, TEST_IMAGE_LIST, BATCH_SIZE=64):
        """
        Predict the task labels corresponding to the input images
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        test_dataset = ChestXrayDataSetTest(image_list_file=TEST_IMAGE_LIST,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]),
                                        combine_pneumonia=self.combine_pneumonia)
        if self.distributed:
            sampler = DistributedSampler(test_dataset)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=8, pin_memory=True,
                                sampler=sampler)
        else:
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=8, pin_memory=True)

        # initialize the ground truth and output tensor
        gt = torch.FloatTensor().cuda()
        pred = torch.FloatTensor().cuda()

        # switch to evaluate mode
        self.net.eval()

        for i, (inputs, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # inputs = inputs.to(self.device)
            # target = target.to(self.device)
            inputs = inputs.cuda()
            target = target.cuda()
            gt = torch.cat((gt, target), 0)

            # Shape of input == [BATCH_SIZE, NUM_CROPS=19, CHANNELS=3, HEIGHT=224, WIDTH=244]
            bs, n_crops, c, h, w = inputs.size()
            inputs = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)

            # Pass through the network and take average prediction from all the crops
            output = self.net(inputs)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)
        gt = gt.cpu().numpy()
        pred = pred.cpu().numpy()

        return gt, pred

    def evaluate(self, TEST_IMAGE_LIST, BATCH_SIZE=64, cm_path='cm', roc_path='roc'):
        """
        Evaluate on the test set plotting confusion matrix and roc curves
        """
        gt, pred = self.predict(TEST_IMAGE_LIST, BATCH_SIZE=64)
        print (pred)

        # Compute ROC scores
        labels = ['Normal', 'Bacterial', 'Viral', 'COVID-19']
        if self.combine_pneumonia:
            labels = ['Normal', 'Pneumonia', 'COVID-19']
        self.compute_AUC_scores(gt, pred, labels)

        # Plot ROC scores
        self.plot_ROC_curve(gt, pred, labels, roc_path)

        # Treat the max. output as prediction. 
        # Plot Confusion Matrix
        gt = gt.argmax(axis=1)
        pred = pred.argmax(axis=1)
        self.plot_confusion_matrix(gt, pred, labels, cm_path)

    def F1(self, TEST_DIR, out_file, BATCH_SIZE=64):
        """
        Evaluate on multiple test sets and compute F1 scores
        """
        f = open(out_file, 'w')
        for test_file in glob.glob(os.path.join(TEST_DIR, '*.txt')):
            print (test_file)
            gt, pred = self.predict(test_file, BATCH_SIZE)
            # Treat the max. output as prediction. 
            gt = gt.argmax(axis=1)
            pred = pred.argmax(axis=1)
            f1 = f1_score(gt, pred, average='macro')
            f.write('%s %.6f\n' % (test_file, f1))
        f.close()

    def plot_confusion_matrix(self, y_true, y_pred, labels, cm_path):
        norm_cm = confusion_matrix(y_true, y_pred, normalize='true')
        norm_df_cm = pd.DataFrame(norm_cm, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        sn.heatmap(norm_df_cm, annot=True, fmt='.2f', square=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s_norm.png' % cm_path, pad_inches = 0, bbox_inches='tight')
        
        cm = confusion_matrix(y_true, y_pred)
        # Finding the annotations
        cm = cm.tolist()
        norm_cm = norm_cm.tolist()
        annot = [
            [("%d (%.2f)" % (c, nc)) for c, nc in zip(r, nr)]
            for r, nr in zip(cm, norm_cm)
        ]
        plt.figure(figsize = (10,7))
        sn.heatmap(norm_df_cm, annot=annot, fmt='', cbar=False, square=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s.png' % cm_path, pad_inches = 0, bbox_inches='tight')
        print (cm)

        accuracy = np.sum(y_true == y_pred) / len(y_true)
        print ("Accuracy: %.5f" % accuracy)

    def compute_AUC_scores(self, y_true, y_pred, labels):
        """
        Computes the Area Under the Curve (AUC) from prediction scores

        y_true.shape  = [n_samples, n_classes]
        y_preds.shape = [n_samples, n_classes]
        labels.shape  = [n_classes]
        """
        AUROC_avg = roc_auc_score(y_true, y_pred)
        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
            print('The AUROC of {0:} is {1:.4f}'.format(label, roc_auc_score(y, pred)))

    def plot_ROC_curve(self, y_true, y_pred, labels, roc_path): 
        """
        Plots the ROC curve from prediction scores

        y_true.shape  = [n_samples, n_classes]
        y_preds.shape = [n_samples, n_classes]
        labels.shape  = [n_classes]
        """
        n_classes = len(labels)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
            fpr[label], tpr[label], _ = roc_curve(y, pred)
            roc_auc[label] = auc(fpr[label], tpr[label])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[label] for label in labels]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for label in labels:
            mean_tpr += interp(all_fpr, fpr[label], tpr[label])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.3f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.3f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=2)

        if len(labels) == 4:
            colors = ['green', 'cornflowerblue', 'darkorange', 'darkred']
        else:
            colors = ['green', 'cornflowerblue', 'darkred']
        for label, color in zip(labels, cycle(colors)):
            plt.plot(fpr[label], tpr[label], color=color, lw=lw,
                    label='ROC curve of {0} (area = {1:0.3f})'
                    ''.format(label, roc_auc[label]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s.png' % roc_path, pad_inches = 0, bbox_inches='tight')

    def save_model(self, checkpoint_path, model=None):
        if model is None: model = self.net
        torch.save(model.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path, model=None):
        print ("Loading model from %s" % checkpoint_path)
        if model is None: model = self.net
        model.load_state_dict(torch.load(checkpoint_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int) # For distributed processing
    parser.add_argument("--mode", choices=['train', 'test', 'f1'], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--combine_pneumonia", action='store_true', default=False)
    parser.add_argument("--save", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze", action='store_true', default=False)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--cm_path", type=str, default='plots/cm')
    parser.add_argument("--roc_path", type=str, default='plots/roc')
    parser.add_argument("--gamma", type=float, default=2.0)

    # parser.add_argment("--torch_version", "--tv", choices=["0.3", "new"], default="0.3")
    args = parser.parse_args()

    TRAIN_IMAGE_LIST = './data/train.txt'
    VAL_IMAGE_LIST = './data/val.txt'
    TEST_IMAGE_LIST = './data/test.txt'
    TEST_DIR = './data/samples'

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    if args.local_rank is not None:
        torch.distributed.init_process_group(backend='nccl')

    trainer = Trainer(local_rank=args.local_rank, checkpoint=args.checkpoint, combine_pneumonia=args.combine_pneumonia)
    if args.mode == 'test':
        trainer.evaluate(TEST_IMAGE_LIST, cm_path=args.cm_path, roc_path=args.roc_path)
    elif args.mode == 'train':
        assert args.save is not None
        trainer.train(TRAIN_IMAGE_LIST, VAL_IMAGE_LIST, BATCH_SIZE=args.bs, NUM_EPOCHS=300, LR=args.lr,
                        start_epoch=args.start, save_path=args.save, freeze_feature_layers=args.freeze, gamma=args.gamma)
    else:
        trainer.F1(TEST_DIR, 'models/samples.txt')

# Run command for distributed
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 OUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of our training script)
# python -m torch.distributed.launch --nproc_per_node=2 trainer.py --mode train
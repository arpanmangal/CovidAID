"""
Trainer for training and testing the networks
"""

import os
import numpy as np
import datetime
import json
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from read_data import ChestXrayDataSet, load_single_image
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from covidxnet import CovidXNet
from tqdm import tqdm

class Trainer:
    def __init__ (self, local_rank=None, checkpoint=None, combine_pneumonia=False):
        """
        Trainer for the CovidXNet
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

        # self.net = CovidXNet().to(self.device)
        self.net = CovidXNet(combine_pneumonia).cuda()
        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net,
                                                            device_ids=[local_rank],
                                                            output_device=local_rank)
        
        # load model
        if checkpoint is not None:
            self.load_model(checkpoint)

    def train(self, TRAIN_IMAGE_LIST, VAL_IMAGE_LIST, NUM_EPOCHS=10, LR=0.001, BATCH_SIZE=64,
                start_epoch=0, logging=True, save_path=None, freeze_feature_layers=True, inc_recall=None):
        """
        Train the CovidXNet
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        train_dataset = ChestXrayDataSet(image_list_file=TRAIN_IMAGE_LIST,
                                        train_time=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]),
                                        recall_class=inc_recall,
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
                                        train_time=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]),
                                        recall_class=inc_recall,
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
                loss = train_dataset.loss(preds, target)  
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
                loss = val_dataset.loss(preds, target) 
                
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

    def single_predict(self, IMAGE_PATH):
        """
        Predict for a single image
        """
        raise NotImplementedError

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
        transform=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda
                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda
                (lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])

        inputs = load_single_image(IMAGE_PATH, transform=transform)
        inputs = inputs.cuda()

        n_crops, c, h, w = inputs.size()
        inputs = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)

        # Pass through the network and take average prediction from all the crops
        pred = self.net(inputs)
        pred = pred.view(n_crops, -1).mean(dim=0)
        pred = pred.cpu().data.numpy()

        print (pred)

    def predict(self, TEST_IMAGE_LIST, BATCH_SIZE=64, cm_path='cm'):
        """
        Predict the task labels corresponding to the input images
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        test_dataset = ChestXrayDataSet(image_list_file=TEST_IMAGE_LIST,
                                        train_time=False,
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

        print (pred)

        # Compute ROC scores
        labels = ['Normal', 'Bacterial', 'Viral', 'COVID-19']
        if self.combine_pneumonia:
            labels = ['Normal', 'Pneumonia', 'COVID-19']
        self.compute_AUC_scores(gt, pred, labels)

        # Treat the max. output as prediction. 
        # Plot Confusion Matrix
        gt = gt.argmax(axis=1)
        pred = pred.argmax(axis=1)
        self.plot_confusion_matrix(gt, pred, labels, cm_path)

    def plot_confusion_matrix(self, y_true, y_pred, labels, cm_path):
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels).astype(int)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='.0f')
        plt.savefig('%s.png' % cm_path)
        print (cm)

        norm_cm = confusion_matrix(y_true, y_pred, normalize='true')
        df_cm = pd.DataFrame(norm_cm, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='.2f')
        plt.savefig('%s_norm.png' % cm_path)
        print (norm_cm)

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
    parser.add_argument("--mode", choices=['train', 'test'], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--combine_pneumonia", action='store_true', default=False)
    parser.add_argument("--save", type=str)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze", action='store_true', default=False)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--inc_recall", type=int, default=None)
    parser.add_argument("--cm_path", type=str, default='plots/cm')
    # parser.add_argment("--torch_version", "--tv", choices=["0.3", "new"], default="0.3")
    args = parser.parse_args()

    TRAIN_IMAGE_LIST = './data/train.txt'
    VAL_IMAGE_LIST = './data/val.txt'
    TEST_IMAGE_LIST = './data/test.txt'

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    if args.local_rank is not None:
        torch.distributed.init_process_group(backend='nccl')

    trainer = Trainer(local_rank=args.local_rank, checkpoint=args.checkpoint, combine_pneumonia=args.combine_pneumonia)
    if args.mode == 'test':
        trainer.predict(TEST_IMAGE_LIST, cm_path=args.cm_path)
    else:
        assert args.save is not None
        trainer.train(TRAIN_IMAGE_LIST, VAL_IMAGE_LIST, BATCH_SIZE=args.bs, NUM_EPOCHS=300, LR=args.lr,
                        start_epoch=args.start, save_path=args.save, freeze_feature_layers=args.freeze,
                        inc_recall=args.inc_recall)

# Run command for distributed
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=1234 OUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other arguments of our training script)
# python -m torch.distributed.launch --nproc_per_node=2 trainer.py --mode train
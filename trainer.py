"""
Trainer for training and testing the networks
"""

import os
import numpy as np
import datetime
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from covxnet import CovXNet
from tqdm import tqdm

class Trainer:
    def __init__ (self):
        """
        Trainer for the CovXNet
        """
        self.net = CovXNet()
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.net = self.net.cuda()

    def train(self, TRAIN_IMAGE_LIST, VAL_IMAGE_LIST, NUM_EPOCHS=10, LR=0.001, BATCH_SIZE=64, DECAY=5000, logging=True, log_file=None):
        """
        Train the CovXNet
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        train_dataset = ChestXrayDataSet(image_list_file=TRAIN_IMAGE_LIST,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]))
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
                                        ]))
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=8, pin_memory=True)

        for epoch in range(NUM_EPOCHS):
            optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=0.9)

            # switch to train mode
            self.net.train()
            tot_loss = 0.0
            for i, (inputs, target) in tqdm(enumerate(train_loader), total=len(train_dataset)):
                if self.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()

                # Shape of input == [BATCH_SIZE, NUM_CROPS=19, CHANNELS=3, HEIGHT=224, WIDTH=244]
                bs, n_crops, c, h, w = inputs.size()
                # print (inputs.size())
                # print (target.size())
                # break
                inputs = torch.autograd.Variable(inputs.view(-1, c, h, w))
                target = torch.autograd.Variable(target)

                preds = self.net(inputs).view(bs, n_crops, -1).mean(1)
                # print (type(preds))
                # print (preds.shape)
                # print (preds)
                # print (target)

                preds[:, 3] = preds[:, 3] * 10
                target[:, 3] = target[:, 3] * 10

                loss = torch.sum(torch.abs(preds - target) ** 2)
                
                tot_loss += float(loss.data)
                # loss = loss_fn (preds, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss /= len(train_dataset)

            if self.use_cuda:
                # Clear cache
                torch.cuda.empty_cache()

            # Running on validation set
            self.net.eval()
            val_loss = 0.0
            for i, (inputs, target) in tqdm(enumerate(train_loader), total=len(val_dataset)):
                if self.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()

                # Shape of input == [BATCH_SIZE, NUM_CROPS=19, CHANNELS=3, HEIGHT=224, WIDTH=244]
                bs, n_crops, c, h, w = inputs.size()
                inputs = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)
                target = torch.autograd.Variable(target, volatile=True)

                preds = self.net(inputs).view(bs, n_crops, -1).mean(1)
                loss = torch.sum(torch.abs(preds - target) ** 2)
                
                val_loss += float(loss.data)

            val_loss /= len(val_dataset)

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

            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write("{}\n".format(log))

            self.save_model('models/epoch_%d.pth' % (epoch + 1))
            
        print ('Finished Training')

    def predict(self, TEST_IMAGE_LIST, BATCH_SIZE=64):
        """
        Predict the task labels corresponding to the input images
        """
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

        test_dataset = ChestXrayDataSet(image_list_file=TEST_IMAGE_LIST,
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.TenCrop(224),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                            transforms.Lambda
                                            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                        ]))
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=8, pin_memory=True)


        # initialize the ground truth and output tensor
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        if self.use_cuda:
            gt = gt.cuda()
            pred = pred.cuda()

        # switch to evaluate mode
        self.net.eval()

        for i, (inp, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if self.use_cuda:
                inp = inp.cuda()
                target = target.cuda()
            gt = torch.cat((gt, target), 0)

            # Shape of input == [BATCH_SIZE, NUM_CROPS=19, CHANNELS=3, HEIGHT=224, WIDTH=244]
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w), volatile=True)

            # Pass through the network and take average prediction from all the crops
            output = self.net(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

            break

        gt = gt.cpu().numpy().argmax(axis=1)
        pred = pred.cpu().numpy().argmax(axis=1)

        self.plot_confusion_matrix(gt, pred)

    def plot_confusion_matrix(self, y_true, y_pred):
        labels = ['Normal', 'Bacterial', 'Viral', 'COVID']

        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels).astype(int)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='.0f')
        plt.savefig('cm.png')
        print (cm)

        norm_cm = confusion_matrix(y_true, y_pred, normalize='true')
        df_cm = pd.DataFrame(norm_cm, index=labels, columns=labels)
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, fmt='.2f')
        plt.savefig('cm_norm.png')
        print (norm_cm)

    # def score(self, X, Y):
    #     """
    #     Score the model -- compute accuracy
    #     """
    #     pred = self.predict(X)
    #     acc = np.sum(pred == Y) / len(Y)
    #     return float(acc)

    def save_model(self, checkpoint_path, model=None):
        if model is None: model = self.net
        torch.save(model.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path, model=None):
        if model is None: model = self.net
        if self.use_cuda:
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))


if __name__ == '__main__':
    trainer = Trainer()
    TRAIN_IMAGE_LIST = './data/val.txt'
    VAL_IMAGE_LIST = './data/val.txt'
    TEST_IMAGE_LIST = './data/test.txt'
    trainer.load_model('covxnet_transfered.pth.tar')
    # trainer.predict(TEST_IMAGE_LIST)
    trainer.train(TRAIN_IMAGE_LIST, VAL_IMAGE_LIST, BATCH_SIZE=8, NUM_EPOCHS=3)


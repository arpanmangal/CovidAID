"""
Trainer for training and testing the networks
"""

from covxnet import CovXNet
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__ (self):
        """
        Trainer for the CovXNet
        """
        self.net = CovXNet()
        
        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.net = self.net.cuda()

    def train(self, X, Y, epochs=10, lr=0.001, batch_size=64, decay=5000, logging=True, log_file=None):
        """
        Train the CNN

        Params:
        @X: Training data - input of the model
        @Y: Training labels
        @logging: True for printing the training progress after each epoch
        @log_file: Path of log file
        """
        X = [x.reshape(1, self.imgsize, self.imgsize) for x in X]
        
        inputs = torch.FloatTensor(X)
        labels = torch.LongTensor(Y)
        
        train_dataset = TensorDataset(inputs, labels)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.net.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1): # loop over data multiple times
            # Decreasing the learning rate
            if (epoch % decay == 0):
                lr /= 3
                
            optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
            
            tot_loss = 0.0
            for data in tqdm(trainloader):
                # get the inputs
                inputs, labels = data
                if self.cuda_flag:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                o = self.net(inputs)
                loss = criterion(o, labels)

                loss.backward()
                optimizer.step()
                
                tot_loss += loss.item()
                
            tot_loss /= len(trainloader)

            # logging statistics
            timestamp = str(datetime.datetime.now()).split('.')[0]
            log = json.dumps({
                'timestamp': timestamp,
                'epoch': epoch,
                'loss': float('%.7f' % tot_loss),
                'lr': float('%.6f' % lr)
            })
            if logging:
                print (log)

            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write("{}\n".format(log))
            
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
        if self.cuda_flag:
            gt = gt.cuda()
            pred = pred.cuda()

        # switch to evaluate mode
        self.net.eval()

        for i, (inp, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            output = self.net(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

            # break

        gt = gt.cpu().numpy().argmax(axis=1)
        pred = pred.cpu().numpy().argmax(axis=1)

        self.plot_confusion_matrix(gt, pred)

        # inputs = [x.reshape(1, self.imgsize, self.imgsize) for x in inputs]
        # inputs = torch.FloatTensor(inputs)
        # if self.cuda_flag:
        #     inputs = inputs.cuda()

        # self.net.eval()
        # with torch.no_grad():
        #     labels = self.net(inputs).cpu().numpy()
            
        # return np.argmax(labels, axis=1)

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
        if self.cuda_flag:
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))


if __name__ == '__main__':
    trainer = Trainer()
    TEST_IMAGE_LIST = './data/test.txt'
    trainer.load_model('covxnet_transfered.pth.tar')
    trainer.predict(TEST_IMAGE_LIST)

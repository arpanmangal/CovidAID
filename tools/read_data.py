# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

class ChestXrayDataSet(Dataset):
    def __init__(self, image_list_file, train_time=True, transform=None, recall_class=None, combine_pneumonia=False):
        """
        Create the Data Loader.
        Since class 3 (Covid) has limited data, dataset size will be accordingly at train time.
        Code is written in generic form to assume last class as the rare class

        Args:
            image_list_file: path to the file containing images
                with corresponding labels.
            train_time: True/False
            transform: optional transform to be applied on a sample.
            recall_class: Integer representing class whose recall to increase.
            combine_pneumonia: True for combining Baterial and Viral Pneumonias into one class
        """
        self.NUM_CLASSES = 3 if combine_pneumonia else 4
        self.recall_class = recall_class

        if recall_class is not None:
            raise ValueError("Depretiated Feature")
            assert 0 <= recall_class < self.NUM_CLASSES
            print ("Increasing the recall of class %d" % recall_class)

        # Set of images for each class
        image_names = [[] for _ in range(self.NUM_CLASSES)]

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = int(items[1])
                image_names[label].append(image_name)

        self.train_time = train_time
        self.image_names = image_names
        self.transform = transform

        label_dist = [len(cnames) for cnames in image_names]

        # Number of images of each class desired
        self.num_covid = int(label_dist[-1])
        covid_factor = 5.0
        self.num_normal = int(self.num_covid * covid_factor)

        if combine_pneumonia:
            self.num_pneumonia = int(self.num_covid * covid_factor)
            self.total = self.num_covid + self.num_pneumonia + self.num_normal
            self.loss_weight_minus = torch.FloatTensor([self.num_normal, self.num_pneumonia, self.num_covid]).unsqueeze(0).cuda() / self.total
            self.loss_weight_plus = 1.0 - self.loss_weight_minus
        else:
            self.num_viral = int(self.num_covid * covid_factor)
            self.num_bact = int(self.num_covid * covid_factor)
            self.total = self.num_covid + self.num_viral + self.num_bact + self.num_normal
            self.loss_weight_minus = torch.FloatTensor([self.num_normal, self.num_bact, self.num_viral, self.num_covid]).unsqueeze(0).cuda() / self.total
            self.loss_weight_plus = 1.0 - self.loss_weight_minus

        print (self.loss_weight_plus, self.loss_weight_minus)

        if self.train_time:
            if combine_pneumonia:
                self.partitions = [self.num_covid,
                                    self.num_covid + self.num_normal,
                                    self.num_covid + self.num_normal + self.num_pneumonia]
            else:
                self.partitions = [self.num_covid,
                                    self.num_covid + self.num_normal,
                                    self.num_covid + self.num_normal + self.num_bact,
                                    self.num_covid + self.num_normal + self.num_bact + self.num_viral]
        else:
            self.partitions = [len(image_names[-1])] # Set of COVID image names
            for l in range(0, self.NUM_CLASSES - 1):
                self.partitions.append(self.partitions[-1] + len(image_names[l]))

        assert len(self.partitions) == self.NUM_CLASSES

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """

        def __one_hot_encode(l):
            v = [0] * self.NUM_CLASSES
            v[l] = 1
            if self.recall_class is not None and l == self.recall_class:
                v = [-0.5] * self.NUM_CLASSES
                v[l] = 1.5
            return v

        image_name = None
        # print (index, self.partitions, len(self), sum([len(cnames) for cnames in self.image_names]))
        if index < self.partitions[0]:
            # Return a covid image
            data_idx = index
            image_name = self.image_names[self.NUM_CLASSES - 1][data_idx]
            label = __one_hot_encode(self.NUM_CLASSES - 1)
        else:
            # Return non-covid image
            for l in range(1, self.NUM_CLASSES):
                if index < self.partitions[l]:
                    class_idx = l - 1
                    label = __one_hot_encode(class_idx)
                    if self.train_time:
                        # Return a random image
                        image_name = random.choice(self.image_names[class_idx])
                    else:
                        # Return the exact needed image
                        data_idx = index - self.partitions[l - 1]
                        image_name = self.image_names[class_idx][data_idx]
                    break

        assert image_name is not None

        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return self.partitions[-1]

    def loss(self, output, target):
        """
        Binary weighted cross-entropy loss for each class
        """
        weight_plus = torch.autograd.Variable(self.loss_weight_plus.repeat(1, target.size(0)).view(-1, self.loss_weight_plus.size(1)).cuda())
        weight_neg = torch.autograd.Variable(self.loss_weight_minus.repeat(1, target.size(0)).view(-1, self.loss_weight_minus.size(1)).cuda())

        loss = output
        pmask = (target >= 0.5).data
        nmask = (target < 0.5).data
        
        epsilon = 1e-15
        loss[pmask] = (loss[pmask] + epsilon).log() * weight_plus[pmask]
        loss[nmask] = (1-loss[nmask] + epsilon).log() * weight_plus[nmask]
        loss = -loss.sum()
        return loss / target.size(0)

def load_single_image(img_path, transform=None):
    """
    Load a single image for inference
    """
    image = Image.open(img_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    return image

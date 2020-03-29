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
    def __init__(self, image_list_file, train_time=True, transform=None):
        """
        Create the Data Loader.
        Since class 3 (Covid) has limited data, dataset size will be accordingly at train time.
        Code is written in generic form to assume last class as the rare class

        Args:
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.NUM_CLASSES = 4

        # Set of images for each class
        image_names = [[] for _ in range(self.NUM_CLASSES)]

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                # label = __one_hot_encode(int(items[1]))
                label = int(items[1])
                image_names[label].append(image_name)

        self.train_time = train_time
        self.image_names = image_names
        self.transform = transform

        label_dist = [len(cnames) for cnames in image_names]

        # Number of images of each class desired
        self.num_covid = int(label_dist[3])
        self.num_viral = int(self.num_covid * 1.5)
        self.num_bact = int(self.num_covid * 2.0)
        self.num_normal = int(self.num_covid * 1.5)
        self.total = self.num_covid + self.num_viral + self.num_bact + self.num_normal

        if self.train_time:
            self.partitions = [self.num_covid,
                                self.num_covid + self.num_normal,
                                self.num_covid + self.num_normal + self.num_bact,
                                self.num_covid + self.num_normal + self.num_bact + self.num_viral]
        else:
            self.partitions = [len(image_names[-1])]
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
        # if self.train_time:
        #     return self.total
        # else:
        #     return sum([len(cnames) for cnames in self.image_names])

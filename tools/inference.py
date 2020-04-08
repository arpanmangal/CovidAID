"""
Script to process set of images and output predictions
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import argparse
from covidxnet import CovidXNet
from tqdm import tqdm
import termtables as tt


class CovidDataLoader(Dataset):
    """
    Read images and corresponding labels.
    """
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: path to image directory.
            transform: optional transform to be applied on a sample.
        """
        self.image_names = [img for img in glob.glob(os.path.join(image_dir, '*'))]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its name
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name.split('/')[-1]

    def __len__(self):
        return len(self.image_names)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--combine_pneumonia", action='store_true', default=False)
    args = parser.parse_args()

    # Load the model
    model = CovidXNet(args.combine_pneumonia).cuda()
    model.load_state_dict(torch.load(args.checkpoint))

    # Load the data
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    test_dataset = CovidDataLoader(image_dir=args.img_dir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda
                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda
                (lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=64,
                    shuffle=False, num_workers=8, pin_memory=True)

    # initialize the output tensor
    pred = torch.FloatTensor().cuda()
    pred_names = []

    # switch to evaluate mode
    model.eval()

    for i, (inputs, names) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs = inputs.cuda()

        # Shape of input == [BATCH_SIZE, NUM_CROPS=10, CHANNELS=3, HEIGHT=224, WIDTH=244]
        bs, n_crops, c, h, w = inputs.size()
        inputs = torch.autograd.Variable(inputs.view(-1, c, h, w), volatile=True)

        # Pass through the network and take average prediction from all the crops
        output = model(inputs)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)
        pred_names += names

    pred = pred.cpu().numpy()

    # print (pred, type(pred))
    # print (names, type(names))
    
    assert len(pred) == len(names)

    scores = []
    for p, n in zip(pred, names):
        p = ["%.1f %%" % (i * 100) for i in p]
        scores.append([n] + p)

    header=['Name', 'Normal', 'Bacterial', 'Viral', 'COVID-19']
    alignment="c"*5
    if args.combine_pneumonia:
        header = ['Name', 'Normal', 'Pneumonia', 'COVID-19']
        alignment = "c"*4

    string = tt.to_string(
        scores,
        header=header,
        style=tt.styles.ascii_thin_double,
        padding=(0, 1),
        alignment=alignment
    )

    print (string)
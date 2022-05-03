"""
Code to transfer weights from CheXNet (torch 1.0) to CovidAID
""" 

import sys
from covidaid import CovidAID, CheXNet
import torch
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--combine_pneumonia", action='store_true', default=False)
parser.add_argument("--chexnet_model_checkpoint", "--old", type=str, default="./data/CheXNet_model.pth.tar")
parser.add_argument("--covidaid_model_trained_checkpoint", "--new", type=str, default="./models/CovidAID_transfered.pth.tar")
args = parser.parse_args()

chexnet_model_checkpoint = args.chexnet_model_checkpoint
covidaid_model_trained_checkpoint = args.covidaid_model_trained_checkpoint

model = CovidAID(combine_pneumonia=args.combine_pneumonia)

def load_weights(checkpoint_pth, state_dict=True):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_pth)
    else:
        checkpoint = torch.load(checkpoint_pth,map_location=torch.device('cuda:0'))
    
    if state_dict:
        #For compatibility of latest torch versions
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return checkpoint['state_dict']
    else:
        return model

def get_top_keys(model, depth=0):
    return set({w.split('.')[depth] for w in model.keys()})

chexnet_model = load_weights(chexnet_model_checkpoint)
template = model.state_dict()

assert get_top_keys(chexnet_model, depth=2) == set({'features', 'classifier'})
assert get_top_keys(template, depth=1) == set({'features', 'classifier'})

# print (chexnet_model.keys())
# print (template.keys())
# print (model.state_dict().keys())

c_keys = {k for k in chexnet_model.keys()}
t_keys = {'module.' + k for k in template.keys()}

assert len(c_keys.difference(t_keys)) == 0

#This will error for "*.num_batches_tracked" layers
#assert len(t_keys.difference(c_keys)) == 0


# Transfer the feature weights
for k, w in template.items():
    chex_key = 'module.' + k

    if k.split('.')[1] == 'classifier':
        # Uncomment below to copy trained weights of pneumonia
        # 6th class is pneumonia in CheXNet => Copy it's weights to pneumonia classes
        # for c in [1, 2, 3]:
        #     template[k][c, ...] = chexnet_model[chex_key][6, ...]
        print ('doing nothing for', k)
    else:
        # print (type(template[k]), template[k].size())
        # print (type(chexnet_model[chex_key]), chexnet_model[chex_key].size())
        if not chex_key.endswith('.num_batches_tracked'):
            assert chexnet_model[chex_key].size() == template[k].size()
            template[k] = chexnet_model[chex_key]

torch.save(template, covidaid_model_trained_checkpoint)

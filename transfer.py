"""
Code to transfer weights from CheXNet (torch 0.3) to CovXNet (torch >=1.2)
"""

from covxnet import CovXNet, CheXNet
import torch


chexnet_model_checkpoint = "model.pth.tar"
covxnet_model_checkpoint = "covxnet_blank.pth.tar"
covxnet_model_trained_checkpoint = "covxnet_transfered.pth.tar"

model = CovXNet()
torch.save(model.state_dict(), covxnet_model_checkpoint)

def load_weights(checkpoint_pth, state_dict=True):
    if torch.cuda.is_available():
        model = torch.load(checkpoint_pth)
    else:
        model = torch.load(checkpoint_pth, map_location=torch.device('cpu'))
    
    if state_dict:
        return model['state_dict']
    else:
        return model

def get_top_keys(model, depth=0):
    return set({w.split('.')[depth] for w in model.keys()})

chexnet_model = load_weights(chexnet_model_checkpoint)
template = load_weights(covxnet_model_checkpoint, state_dict=False)

assert get_top_keys(chexnet_model, depth=2) == set({'features', 'classifier'})
assert get_top_keys(template, depth=1) == set({'features', 'classifier'})

# print (chexnet_model.keys())
# print (template.keys())
# print (model.state_dict().keys())

c_keys = {k for k in chexnet_model.keys()}
t_keys = {'module.' + k for k in template.keys()}

print (c_keys.difference(t_keys))
print (t_keys.difference(c_keys))


# Transfer the feature weights
for k, w in template.items():
    chex_key = 'module.' + k

    if k.split('.')[1] == 'classifier':
        # 6th class is pneumonia in CheXNet => Copy it's weights to pneumonia classes
        for c in [1, 2, 3]:
            template[k][c, ...] = chexnet_model[chex_key][6, ...]
    else:
        # print (type(template[k]), template[k].size())
        # print (type(chexnet_model[chex_key]), chexnet_model[chex_key].size())
        assert chexnet_model[chex_key].size() == template[k].size()
        template[k] = chexnet_model[chex_key]

torch.save(template, covxnet_model_trained_checkpoint)

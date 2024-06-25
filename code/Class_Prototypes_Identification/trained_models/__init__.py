from .clip_backbone import *
import os
from .plip import *
import torch.nn as nn
from torchvision import models


def get_model(encoder, weights_dir=None):
    if encoder == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        descriptors_dimension = 512

    elif encoder == "resnet18_SSL":
        # https://github.com/ozanciga/self-supervised-histopathology
        model_path = os.path.join(weights_dir, 'tenpercent_resnet18.ckpt')
        state_dict = torch.load(model_path, map_location="cpu")['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model = models.__dict__['resnet18'](pretrained=False)
        model_dict = model.state_dict()
        weights = {k: v for k, v in state_dict.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        else:
            print('loading weights from tenpercent_resnet18.ckpt')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        print("=> loaded pre-trained model '{}'".format(model_path))
        model.fc = torch.nn.Identity()
        descriptors_dimension = 512

    elif encoder == "clip":
        model = clip_backbone.image_encoder()
        descriptors_dimension = 512

    elif encoder == "plip":
        model = plip.image_encoder()
        descriptors_dimension = 512

    else:
        raise NotImplementedError

    return model, descriptors_dimension

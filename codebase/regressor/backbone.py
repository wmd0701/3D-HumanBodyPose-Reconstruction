from torchvision import models
from torch import nn

def load_backbone(backbone):

    if backbone == 'resnet50':
        model = load_resnet50()
        model.fc = nn.Identity()
        return model
    else:
        raise Exception(f'Backbone \'{backbone}\' is not supported.')


def load_resnet50(freeze=True):
    resnet50 = models.resnet50(pretrained=True)
    
    for param in resnet50.parameters():
        param.requires_grad = (not freeze)
    
    return resnet50
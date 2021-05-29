from torchvision import models

def load_resnet(freeze=True):
    resnet50 = models.resnet50(pretrained=True)
    
    for param in resnet50.parameters():
        param.requires_grad = not freeze
    
    return resnet50
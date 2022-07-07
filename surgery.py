import torch

import timm


from timm.models.resnet import Bottleneck, make_blocks

if __name__ == "__main__":
    model = timm.create_model("resnet152",
                              pretrained=False,
                              num_classes=0)
    
    if True:
        batch_size = 2
        input_size = 32
        
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)
        model.train()
        model(dummy_input)
    
    
    
    
    
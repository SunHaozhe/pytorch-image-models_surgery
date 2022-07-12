import torch

import timm


from timm.models.resnet import Bottleneck, make_blocks


if __name__ == "__main__":
    model_name = "resnet152"
    model = timm.create_model(model_name,
                              pretrained=False,
                              num_classes=0)
    
    if False:
        batch_size = 1
        input_size = 32
        
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)
        model.train()
        
        model(dummy_input)
    
    
    
    
    
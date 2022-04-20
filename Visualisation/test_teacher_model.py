from timesformer.datasets import loader
from timesformer.utils.parser import load_config, parse_args
import numpy as np
import cv2
from timesformer.models import build_model
import torchvision


class TransformerFeatureMap:
    def __init__(self, model, layer_name='model.norm'):
        self.model = model
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_feature)
        self.feature = []

    def get_feature(self, module, input, output):
        self.feature.append(output.cpu())

    def __call__(self, input_tensor):
        self.feature = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return self.feature

def loader_data():
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)
    train_loader = loader.construct_loader(cfg, "val")
    return train_loader

def loader_model():
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)
    model = build_model(cfg)
    return model

def show_mask_on_image(img, mask):
    img = np.float32(img)
    img = (img - np.min(img))/5.15
    cam = 0 + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_heatmap(mask):
    #print(np.min(mask),np.max(mask))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)
    heatmap = np.float32(heatmap) / 255
    
    return np.uint8(255 * heatmap)
    


def main():
    from pathlib import Path
    import torch
    from timesformer.models.vit import TimeSformer
    from vit_rollout_Time import VITAttentionRollout_t
    import numpy as np
    import cv2
    from matplotlib import rc
    import matplotlib.pyplot as plt
    import os
    import torch.nn as nn
    rc('animation', html='jshtml')

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    model_name="r2plus1d_18"
    pretrained=True
    weights="/AS_Neda/timeSformer/pretrained/checkpoint_paper.pt"
    device = None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.fc.bias.data[0] = 55.6

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])
        print("weights loaded")
    
    minloss = 100
    maxloss = 0
    total = 0
    n = 0
    train = loader_data()
    loss_fun = nn.MSELoss()
    for (inputs, labels, _, meta) in train:
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        preds = model(inputs)
        loss = loss_fun(preds.squeeze(0),labels)
        #print("preds,labels,loss",preds.squeeze(0),labels,loss)
        if loss<minloss:
            minloss = loss.cpu()
        if loss>maxloss:
            maxloss = loss.cpu()
        # total = total+loss.cpu()
        # n = n+1
    print(minloss,maxloss)  
    print("average",total/n)



if __name__ == "__main__":
    main()

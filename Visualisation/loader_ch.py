from timesformer.datasets import loader
from timesformer.utils.parser import load_config, parse_args
import numpy as np
import cv2
from timesformer.models import build_model


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
    train_loader = loader.construct_loader(cfg, "train")
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
    #mask = (mask - np.min(mask))/(np.max(mask) - np.min(mask))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HOT)
    heatmap = np.float32(heatmap) / 255
    #img = img[0].transpose((1,2,0))
    #heatmap = mask
    #print(np.min(mask),np.max(mask))
    cam = heatmap + 0.7*img
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
    from vit_rollout import VITAttentionRollout
    from vit_rollout_Time import VITAttentionRollout_t
    import numpy as np
    import cv2
    from matplotlib import rc
    import matplotlib.pyplot as plt
    import os
    rc('animation', html='jshtml')

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import FuncAnimation, PillowWriter
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    model = TimeSformer(img_size=112, num_classes=1, num_frames=32, attention_type='divided_space_time',  pretrained_model='/AS_Neda/timeSformer/pretrained/models/checkpoint_epoch_00064.pyth')
    #model = loader_model()
    train = loader_data()
    (inputs, (EF,EDV,ESV), _, meta) = next(iter(train))
    print(EF,ESV,EDV)
#     heatmap_total = []
#     ratio = 0.95
#     attention_rollout = VITAttentionRollout(model, head_fusion='min', 
#     discard_ratio=ratio)
#     ine = inputs[0].unsqueeze(0)
#     mask = attention_rollout(ine)
#     np_img = np.array(ine)
#     temporal = np.max(mask,axis =(1,2))
#     fig = plt.figure(figsize = (10,15))
#     x = range(32)
#     plt.bar(x,temporal/np.max(temporal))
#     plt.show()
#     plt.savefig('time.png')
#     # plt.imshow(mask[2])
#     # plt.savefig('mask.png')
#     heatmap = []
#     heatmap_alone = []
#     aside = np.array([])
#     for i in range(len(mask)):
#         mask_reshape = cv2.resize(mask[i],(np_img.shape[3], np_img.shape[4]))
#         print(np.min(mask_reshape),np.max(mask_reshape))
#         mask_reshape = np.array(mask_reshape)
#         heatmap.append(show_mask_on_image(ine[0,:,i,:,:].permute(1,2,0), mask_reshape))
#         heatmap_alone.append(mask_reshape)
#         if len(aside) == 0:
#             aside = show_mask_on_image(ine[0,:,i,:,:].permute(1,2,0), mask_reshape)
#         else:
#             if i%4 == 0:
#                 aside = np.hstack((aside,show_mask_on_image(ine[0,:,i,:,:].permute(1,2,0), mask_reshape)))
    
#     # heatmap_alone = np.array(heatmap_alone)
#     # fig, ax = plt.subplots()
#     # frames = [[ax.imshow(heatmap_alone[i])] for i in range(len(heatmap_alone))]
#     # ani = animation.ArtistAnimation(fig, frames)
#     # f = r"/AS_Neda/timeSformer/runs/video_heat"+str(ratio)+".gif" 
#     # writergif = animation.PillowWriter(fps=10) 
#     # ani.save(f, writer=writergif)
#     fig = plt.figure(figsize = (10,3))
#     plt.imshow(aside)
#     plt.axis('off')
#     plt.colorbar(label="Attention", orientation="horizontal",fraction=0.06)
#     plt.show()
#     plt.savefig('hstack.png')
    
#     heatmap = np.array(heatmap)
#     fig, ax = plt.subplots()
#     frames = [[ax.imshow(heatmap[i])] for i in range(len(heatmap))]
#     ani = animation.ArtistAnimation(fig, frames)
#     f = r"/AS_Neda/timeSformer/runs/video"+str(ratio)+".gif" 
#     writergif = animation.PillowWriter(fps=10) 
#     ani.save(f, writer=writergif)


if __name__ == "__main__":
    main()

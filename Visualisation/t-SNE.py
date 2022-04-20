from timesformer.datasets import loader
from timesformer.utils.parser import load_config, parse_args
import numpy as np
import cv2
import torch
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt

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
            output = self.model(input_tensor.cuda())

        return self.feature


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range

    
def loader_data():
    args = parse_args()
    if args.num_shards > 1:
        args.output_dir = str(args.job_dir)
    cfg = load_config(args)
    train_loader = loader.construct_loader(cfg, "train")
    return train_loader

def show_mask_on_image(img, mask):
    img = np.float32(img) 
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    from pathlib import Path
    import torch
    from timesformer.models.vit import TimeSformer
    from vit_rollout import VITAttentionRollout
    import numpy as np
    import cv2
    from matplotlib import rc
    rc('animation', html='jshtml')

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    model = TimeSformer(img_size=112, num_classes=1, num_frames=8, attention_type='divided_space_time',  pretrained_model='/AS_Neda/timeSformer/pretrained/models/checkpoint_epoch_00064.pyth')

    loader = loader_data()
    (inputs, labels, _, meta) = next(iter(loader))
    
    feature_extractor = TransformerFeatureMap(model.cuda())
    feature = feature_extractor(inputs)
    features = np.array(feature[0].mean(axis=1))
    labels_tot = np.array(labels)
    print(features.shape)
    for cur_iter, (inputs, labels, _, meta) in enumerate(loader):
        feature = feature_extractor(inputs)
        features = np.concatenate((features,feature[0].mean(axis=1).numpy()))
        labels_tot = np.concatenate((labels_tot,np.array(labels)))

    print('Done importing features',len(labels_tot))
    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # print(np.unique(labels_tot))
    #N = len(np.unique(labels_tot))
    N=6

    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(6,6))

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    #bounds = np.linspace(0,100,5)
    bounds = np.array([0,35,45,55,65,100])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(tx,ty,c=labels_tot,cmap=cmap,norm=norm)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Custom cbar')
    ax.set_title('T-SNE')
    plt.show()
    plt.savefig('scatter_train.png')

        


if __name__ == "__main__":
    main()

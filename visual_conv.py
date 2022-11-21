from __future__ import print_function

import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from vit_pytorch.linform import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import random
from vit_pytorch.efficient import ViT
from utils import create_logger
from vit_pytorch.cross_vit import CrossViT

v = CrossViT(
    image_size = 224,
    num_classes = 9,
    depth = 4,               # number of multi-scale encoding blocks
    sm_dim = 192,            # high res dimension
    sm_patch_size = 32,      # high res patch size (should be smaller than lg_patch_size)
    sm_enc_depth = 2,        # high res depth
    sm_enc_heads = 8,        # high res heads
    sm_enc_mlp_dim = 512,   # high res feedforward dimension
    lg_dim = 384,            # low res dimension
    lg_patch_size = 56,      # low res patch size
    lg_enc_depth = 3,        # low res depth
    lg_enc_heads = 8,        # low res heads
    lg_enc_mlp_dim = 512,   # low res feedforward dimensions
    cross_attn_depth = 2,    # cross attention rounds
    cross_attn_heads = 8,    # cross attention heads
    dropout = 0,
    emb_dropout = 0
)


# Training settings
batch_size = 16
epochs = 20
lr = 1e-4
gamma = 0.9
seed = 42

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)

            x = module(x)
            print(name)
            if self.extracted_layers is None or name in self.extracted_layers and 'fc' not in name:
                outputs[name] = x

        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (256, 256))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

efficient_transformer = Linformer(
    dim=256,
    seq_len=49 +1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64,
    dropout=0
)

model = ViT(
    dim=256,
    image_size=224,
    patch_size=32,
    num_classes=9,
    transformer=efficient_transformer,
    channels=3,
).to('cuda')


if __name__ == '__main__':

    # model = v.to('cuda')
    model.load_state_dict(torch.load(r'./model/10_model.pth'))
    # model = torch.load(r'D:\python_program\CV-vit\vit-pytorch-main\model\backup\10_model.pth')
    # model.eval()
    # criterion = nn.CrossEntropyLoss()
    # model = torch.hub.load('facebookresearch/deit:main',
    #     'deit_tiny_patch16_224', pretrained=True)
    model.eval()

    # if args.use_cuda:
    #     model = model.cuda()

    path_img = r'D:\python_program\CV-vit\vit-pytorch-main\data\train\4_1.mat'
    # path_net = os.path.join(BASE_DIR, "..", "..", "Data", "net_params_72p.pkl")
    # output_dir = os.path.join(BASE_DIR, "..", "..", "Result", "backward_hook_cam")
    input_dict = h5py.File(path_img)
    img = input_dict['output']
    A = torch.complex(torch.from_numpy(img['real'].astype(np.float32)),
                      torch.from_numpy(img['imag'].astype(np.float32)))
    input_tensor = A.view(3, 224, 224)
    # img = A.numpy()
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # img = Image.open(args.image_path)
    # img = img.resize((224, 224))
    # a = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # img = torch.Tensor(img)
    # img = img.permute(2, 0, 1)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to('cuda')


    dst = './feautures'
    therd_size = 256
    myexactor = FeatureExtractor(model, None)
    outs = myexactor(input_tensor)
    for k, v in outs.items():
        features = v[0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')
            if 'fc' in k:
                continue

            feature = features.data.numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, k)

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.png')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            dst_file = os.path.join(dst_path, str(i) + '.png')
            cv2.imwrite(dst_file, feature_img)
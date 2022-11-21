from __future__ import print_function

import glob
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch.linform import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import random
from vit_pytorch.efficient import ViT
# from utils import create_logger
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


batch_size = 16
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42
device = 'cuda'

# os.makedirs('data', exist_ok=True)

# train_dir = 'data/train'
test_dir = 'data/test'

# with zipfile.ZipFile('train.zip') as train_zip:
#     train_zip.extractall('data')
#
# with zipfile.ZipFile('test.zip') as test_zip:
#     test_zip.extractall('data')

# train_list = glob.glob(os.path.join(train_dir, '*.mat'))
test_list = glob.glob(os.path.join(test_dir, '*.mat'))

# print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")


# labels = [path.split('\\')[-1].split('_')[0] for path in train_list]

# random_idx = np.random.randint(1, len(train_list), size=9)

# fig, axes = plt.subplots(3, 3, figsize=(16, 12))
#
# for idx, ax in enumerate(axes.ravel()):
#     img = Image.open(train_list[idx])
#     ax.set_title(labels[idx])
#     ax.imshow(img)

# train_list, valid_list = train_test_split(train_list,
#                                           test_size=0.1,
#                                           stratify=labels,
#                                           random_state=seed)

# print(f"Train Data: {len(train_list)}")
# print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )
#
# val_transforms = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ]
# )
#
# test_transforms = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ]
# )



class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        input_dict = h5py.File(img_path)
        img = input_dict['output']
        A = torch.complex(torch.from_numpy(img['real'].astype(np.float32)),torch.from_numpy(img['imag'].astype(np.float32)))
        # A = torch.stack([torch.from_numpy(img['real'].astype(np.float32)),torch.from_numpy(img['imag'].astype(np.float32))],dim =0)
        img = A.view(3,224,224)
        # A = torch.stack([torch.from_numpy(img['real'].astype(np.float32)),torch.from_numpy(img['imag'].astype(np.float32))],dim =0)
        # img = np.transpose(A, (3,0,1,2))
        # img = Image.open(img_path)
        # img_transformed = self.transform(img)

        label = img_path.split("\\")[-1].split("_")[0]
        label = torch.Tensor(np.array(label).astype(np.float32)).to(torch.int64)-1
        # label = 1 if label == "dog" else 0

        return img, label



# train_data = CatsDogsDataset(train_list, transform=train_transforms)
# valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
# test_data = CatsDogsDataset(test_list, transform=test_transforms)

# train_data = CatsDogsDataset(train_list)
# valid_data = CatsDogsDataset(valid_list)
test_data = CatsDogsDataset(test_list)


# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# print(len(train_data), len(train_loader))

# print(len(valid_data), len(valid_loader))

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
).to(device)

# model = v.to(device)
model.load_state_dict(torch.load(r'D:\python_program\CV-vit\vit-pytorch-main\model\10_model.pth'))
# model = torch.load(r'D:\python_program\CV-vit\vit-pytorch-main\model\backup\10_model.pth')
# model.eval()
# from scipy import io as sio
# sio.savemat('pos.mat',dict({'key':model.pos_embedding.detach().cpu().numpy()}))

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    epoch_val_accuracy = 0
    epoch_val_loss = 0
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        val_output = model(data)
        val_output = torch.abs(val_output)

        # val_logits_mu_real = F.log_softmax(val_output.real, dim=1) + 1j * F.log_softmax(val_output.imag, dim=1)
        # loss_real = F.nll_loss(val_logits_mu_real.real, label)
        # loss_imag = F.nll_loss(val_logits_mu_real.imag, label)
        # val_loss = torch.sqrt(torch.pow(loss_real, 2) + torch.pow(loss_imag, 2))

        # val_logits_mu_real = F.log_softmax(val_output, dim=1)
        # val_loss = F.nll_loss(val_logits_mu_real, label)
        val_loss = criterion(val_output, label)

        # val_accpred = val_logits_mu_real.real + val_logits_mu_real.imag
        acc = (val_output.argmax(dim=1) == label).float().sum()
        epoch_val_accuracy += acc / len(test_list)
        epoch_val_loss += val_loss / len(test_loader)

        print(
            f"Epoch : val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
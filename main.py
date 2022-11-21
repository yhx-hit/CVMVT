from __future__ import print_function

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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

device = 'cuda'

# os.makedirs('data', exist_ok=True)

train_dir = 'data/train'
test_dir = 'data/test'

# with zipfile.ZipFile('train.zip') as train_zip:
#     train_zip.extractall('data')
#
# with zipfile.ZipFile('test.zip') as test_zip:
#     test_zip.extractall('data')

train_list = glob.glob(os.path.join(train_dir, '*.mat'))
test_list = glob.glob(os.path.join(test_dir, '*.mat'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")


labels = [path.split('\\')[-1].split('_')[0] for path in train_list]

random_idx = np.random.randint(1, len(train_list), size=9)

# fig, axes = plt.subplots(3, 3, figsize=(16, 12))
#
# for idx, ax in enumerate(axes.ravel()):
#     img = Image.open(train_list[idx])
#     ax.set_title(labels[idx])
#     ax.imshow(img)

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
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

train_data = CatsDogsDataset(train_list)
valid_data = CatsDogsDataset(valid_list)
test_data = CatsDogsDataset(test_list)


train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))

print(len(valid_data), len(valid_loader))

efficient_transformer = Linformer(
    dim=256,
    seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64,
    dropout=0.
)

# model = ViT(
#     dim=256,
#     image_size=224,
#     patch_size=32,
#     num_classes=9,
#     transformer=efficient_transformer,
#     channels=3,
# ).to(device)

model  = v.to(device)

# loss function
# criterion = nn.NLLLoss()
# criterion = F.nll_loss()
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
logger = create_logger('./log', 'train')
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    stop =0
    for data,label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        output = torch.abs(output)

        # out_logits_mu_real = F.log_softmax(output.real, dim=1)+1j*F.log_softmax(output.imag, dim=1)
        # loss_real = F.nll_loss(out_logits_mu_real.real, label)
        # loss_imag = F.nll_loss(out_logits_mu_real.imag, label)
        # loss = torch.sqrt(torch.pow(loss_real, 2) + torch.pow(loss_imag, 2))

        # out_logits_mu_real = torch.abs(out_logits_mu_real)
        # loss = F.nll_loss(out_logits_mu_real, label)

        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # output_pred = out_logits_mu_real.real+out_logits_mu_real.imag
        # acc = (output_pred.argmax(dim=1) == label).float().sum()

        acc = (output.argmax(dim=1) == label).float().sum()
        epoch_accuracy += acc / len(train_list)
        epoch_loss += loss / len(train_loader)
        stop +=1
        if stop % 1 ==0:
            logger.info(f"Epoch : {epoch + 1} - iteration: {stop,'/',len(train_loader)} - loss : {loss:.4f} - acc: {epoch_accuracy:.4f}")
            # print(
            #     f"Epoch : {epoch + 1} - iteration: {stop/len(train_loader)} - loss : {loss:.4f} - acc: {epoch_accuracy:.4f}"
            # )

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
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

            # val_accpred = val_logits_mu_real.real+val_logits_mu_real.imag
            # acc = (val_accpred.argmax(dim=1) == label).float().sum()

            acc = (val_output.argmax(dim=1) == label).float().sum()
            epoch_val_accuracy += acc / len(valid_list)
            epoch_val_loss += val_loss / len(valid_loader)
    logger.info(
        f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    # print(
    #     f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    # )
    if (epoch+1) % 6 == 0:
        model_dir = os.path.join('./model/', str(epoch+1)+'_cross'+'_model.pth')
        torch.save(model.state_dict(), model_dir)



## test
# dog_probs = []
# model = torch.load(r'D:\python_program\CV-vit\vit-pytorch-main\model\backup\10_model.pth')
# model.eval()
# with torch.no_grad():
#     for data, fileid in test_loader:
#         data = data.to(device)
#         preds = model(data)
#         preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
#         dog_probs += list(zip(list(fileid), preds_list))
#
# dog_probs.sort(key = lambda x : int(x[0]))
# idx = list(map(lambda x: x[0],dog_probs))
# prob = list(map(lambda x: x[1],dog_probs))
# submission = pd.DataFrame({'id':idx,'label':prob})
# submission.to_csv('result.csv',index=False)
#
# ## visualization
# import random
#
# id_list = []
# class_ = {0: 'cat', 1: 'dog'}
#
# fig, axes = plt.subplots(2, 5, figsize=(20, 12), facecolor='w')
#
# for ax in axes.ravel():
#
#     i = random.choice(submission['id'].values)
#
#     label = submission.loc[submission['id'] == i, 'label'].values[0]
#     if label > 0.5:
#         label = 1
#     else:
#         label = 0
#
#     img_path = os.path.join(test_dir, '{}.jpg'.format(i))
#     img = Image.open(img_path)
#
#     ax.set_title(class_[label])
#     ax.imshow(img)
import os
import sys
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import PIL
from datetime import datetime
import h5py

import kornia
from kornia.augmentation.container import AugmentationSequential


import webdataset as wds
from info_nce import InfoNCE
import clip
import pandas as pd
from collections import OrderedDict

from utils import *
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100  # 800
print("device:",device)

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


def main(exp_name):
    # if full_training is True, use large batches and the entire training dataset 
    full_training = True
    # image augmentation just for the CLIP image model that will be more semantic-focused
    train_augs = AugmentationSequential(
        kornia.augmentation.RandomCrop((140, 140), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        data_keys=["input"],
        # random_apply = (1,4)
    )
    print('full_training:',full_training)
    model_name = 'clip_image_vitB' # CLIP ViT-L/14 image embeddings
    print(f"Using model: {model_name}")
    if "resnet" in model_name: 
        clip_extractor = Clipper("RN50")
    else:
        # clip_extractor = Clipper("ViT-L/14", train_transforms=train_augs)
        clip_extractor = Clipper("ViT-B/32", train_transforms=train_augs)
    if "text" in model_name:     
        image_var = 'trial' 
    else:
        image_var = 'images'
    print("image_var =", image_var)
    
    nsd_path = '../../naturalscenesdataset/webdataset/'
    if not full_training: 
        num_devices = 1
        num_workers = 1
        print("num_workers",num_workers)
        batch_size = 16
        print("batch_size",batch_size)
        num_samples = 500 
        global_batch_size = batch_size * num_devices
        print("global_batch_size",global_batch_size)
        num_batches = math.floor(num_samples / global_batch_size)
        num_worker_batches = math.floor(num_batches / num_workers)
        print("num_worker_batches",num_worker_batches)
        train_url = f"{nsd_path}/train/train_subj01_{{0..1}}.tar"
    else:
        num_devices = torch.cuda.device_count()
        print("num_devices",num_devices)
        num_workers = num_devices
        print("num_workers",num_workers)
        batch_size = 300 # 768
        print("batch_size",batch_size)
        num_samples = 24983 # see metadata.json in webdataset_split folder
        global_batch_size = batch_size * num_devices
        print("global_batch_size",global_batch_size)
        num_batches = math.floor(num_samples / batch_size)
        num_worker_batches = math.floor(num_batches / num_workers)
        print("num_worker_batches",num_worker_batches)
        train_url = f"{nsd_path}/train/train_subj01_{{0..49}}.tar"

    train_data = wds.DataPipeline([wds.ResampledShards(train_url),
                        wds.tarfile_to_samples(),
                        wds.shuffle(500,initial=500),
                        wds.decode("torch"),
                        wds.rename(images="jpg;png", voxels="nsdgeneral.npy", embs="sgxl_emb.npy", trial="trial.npy"),
                        wds.to_tuple("voxels", image_var),
                        wds.batched(batch_size, partial=True),
                    ]).with_epoch(num_worker_batches)
    train_dl = wds.WebLoader(train_data, num_workers=num_workers,
                             batch_size=None, shuffle=False, persistent_workers=True)

    # Validation #
    num_samples = 492
    val_batch_size = 300
    num_batches = math.ceil(num_samples / val_batch_size)
    num_worker_batches = math.ceil(num_batches / num_workers)
    print("validation: num_worker_batches",num_worker_batches)

    url = f"{nsd_path}/val/val_subj01_0.tar"
    val_data = wds.DataPipeline([wds.ResampledShards(url),
                        wds.tarfile_to_samples(),
                        wds.decode("torch"),
                        wds.rename(images="jpg;png", voxels="nsdgeneral.npy", 
                                    embs="sgxl_emb.npy", trial="trial.npy"),
                        wds.to_tuple("voxels", image_var),
                        wds.batched(batch_size, partial=True),
                    ]).with_epoch(num_worker_batches)
    val_dl = wds.WebLoader(val_data, num_workers=num_workers,
                           batch_size=None, shuffle=False, persistent_workers=True)
    
    # check that your data loaders are working
    out_dim = 512
    for train_i, (voxel, img_input) in enumerate(train_dl):
        print("idx",train_i)
        print("voxel.shape",voxel.shape)
        if "text" in model_name:
            emb = clip_extractor.embed_curated_annotations(subj01_annots[img_input])
        else:
            emb = clip_extractor.embed_image(img_input)
        print("emb.shape",emb.shape)
        out_dim = emb.shape[1]
        print("out_dim", out_dim)
        break
    
    brain_net = BrainNetwork(out_dim).to(device)
    # brain_net = BrainNetworkLarge(out_dim).to(device)
    no_decay = ['bias']
    opt_grouped_parameters = [
        {'params': [p for n, p in brain_net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in brain_net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    opt = torch.optim.AdamW(opt_grouped_parameters, lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, #3e-4, 
                                                total_steps=EPOCHS*((24983//300)//num_devices), 
                                                final_div_factor=1000,
                                                last_epoch=-1, pct_start=2/EPOCHS)

    nce = InfoNCE()
    using_ddp = False
    #############
    ### Train ###
    #############
    
    epoch = 0
    train_losses = []; val_losses = []
    train_topk = []; val_topk = []
    lrs = []
    epoch_logs = []

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if full_training:
        print(f"Will be saving model checkpoints to checkpoints/{model_name}_{exp_name}_subj01_epoch#.pth")
    else:
        print(f"Warning: not saving model checkpoints")
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    pbar = tqdm(range(epoch, EPOCHS), ncols=250)
    best_val_topk = 0
    for epoch in pbar:
        brain_net.train()
        similarities = []
        inner_bar = tqdm(train_dl)
        for train_i, (voxel, img_input) in enumerate(inner_bar):
            opt.zero_grad()
            voxel = voxel.to(device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    if image_var=='images': # using images
                        emb = clip_extractor.embed_image(img_input)
                    else: # using text captions of the images 
                        emb = clip_extractor.embed_curated_annotations(subj01_annots[img_input])

            emb = emb.float() # cast to float32
            emb_ = brain_net(voxel.float())

            if torch.any(torch.isnan(emb_)):
                raise ValueError("NaN found...")

            emb_ = nn.functional.normalize(emb_,dim=-1) # l2 normalization on the embeddings

            labels = torch.arange(len(emb)).to(device)
            loss_nce = nce(emb_.reshape(len(emb),-1),emb.reshape(len(emb),-1))
            # loss_nce = 0
            # loss_nce = double_nce(emb_.reshape(len(emb),-1),emb.reshape(len(emb),-1))
            loss_soft = soft_clip_loss(emb_.reshape(len(emb),-1), emb.reshape(len(emb),-1))
            loss = loss_nce + loss_soft

            similarities = batchwise_cosine_similarity(emb,emb_)

            percent_correct = topk(similarities,labels,k=1)

            loss.backward()
            opt.step()
            sched.step()

            train_losses.append(loss.item())
            train_topk.append(percent_correct.item())
            
            inner_bar.set_description(f'Soft: {loss_soft:.4f}, NCE: {loss_nce:.4f}, Acc: {percent_correct:.2f}')

        brain_net.eval()    
        for val_i, (val_voxel, val_img_input) in enumerate(val_dl):
            with torch.no_grad(): 
                val_voxel = val_voxel.to(device)

                with torch.cuda.amp.autocast():
                    if image_var=='images': # using images
                        val_emb = clip_extractor.embed_image(val_img_input)
                    else: # using text captions of the images 
                        val_emb = clip_extractor.embed_curated_annotations(subj01_annots[val_img_input])

                    val_emb_ = brain_net(val_voxel)
                    val_emb_ = nn.functional.normalize(val_emb_,dim=-1) # l2 normalization on the embeddings

                    labels = torch.arange(len(val_emb)).to(device)

                    val_loss = nce(val_emb_.reshape(len(val_emb),-1),val_emb.reshape(len(val_emb),-1))

                    val_similarities = batchwise_cosine_similarity(val_emb,val_emb_)

                    val_percent_correct = topk(val_similarities,labels,k=1)

                val_losses.append(val_loss.item())
                val_topk.append(val_percent_correct.item())
                if val_percent_correct > best_val_topk:
                    best_val_topk = val_percent_correct.item()
                    state_dict = brain_net.state_dict()
                    if using_ddp: # if using DDP, convert DDP to non-DDP before saving
                        state_dict = brain_net.module.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': opt.state_dict(),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_topk': train_topk,
                        'val_topk': val_topk,
                        'lrs': lrs,
                        }, f'checkpoints/{model_name}_{exp_name}_subj01_best.pth')

        if epoch%100==99 and full_training:
            print(f'saving checkpoints/{model_name}_{exp_name}_subj01_epoch{epoch+1}.pth...')
            if (not using_ddp) or (using_ddp and local_rank==0):
                state_dict = brain_net.state_dict()
                if using_ddp: # if using DDP, convert DDP to non-DDP before saving
                    state_dict = brain_net.module.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': opt.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_topk': train_topk,
                    'val_topk': val_topk,
                    'lrs': lrs,
                    }, f'checkpoints/{model_name}_{exp_name}_subj01_epoch{epoch+1}.pth')
            if using_ddp:
                dist.barrier() # this tells the other gpus wait for the first gpu to finish saving the model

        lrs.append(opt.param_groups[0]['lr'])

        # logging the average results across batches for current epoch
        logs = OrderedDict(
            loss=np.mean(train_losses[-(train_i+1):]),
            topk=np.mean(train_topk[-(train_i+1):]),
            val_loss=np.mean(val_losses[-(val_i+1):]),
            val_topk=np.mean(val_topk[-(val_i+1):]),
            lr=lrs[-1],
        )
        pbar.set_postfix(**logs)
        epoch_logs.append(logs)
        if full_training:
            pd.DataFrame(epoch_logs).to_csv(f'checkpoints/{model_name}_{exp_name}_subj01.epoch-logs.csv')

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    exp_name = sys.argv[1]
    main(exp_name)
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import PIL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def np_to_Image(x):
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    return (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum


def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.T.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss


def double_nce(preds, targs, temp=0.1):
    # single positive sample from brain-clip same as nce
    # for negative samples, also adds brain-brain similarities
    brain_clip = (preds @ targs.T)/temp
    brain_brain = (preds @ preds.T)/temp
    
    n = brain_brain.shape[-1]
    brain_brain_offd = brain_brain.flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)
    
    # keeps off-diagonal elements only
    logits = torch.concat([brain_clip, brain_brain_offd], dim=-1)  # n, 2n-1
    return F.cross_entropy(logits, torch.arange(n).to(logits.device))


def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0]).to(voxels.device)
    voxels_shuffle = voxels[perm]
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None):
    brain_clip = (preds @ targs.T)/temp
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        return loss
    else:
        return F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
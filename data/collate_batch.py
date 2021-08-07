# encoding: utf-8
# update this three functions make it more readable,
# and add a return varible "img_path" in "val_collate_fn"


import torch


def train_collate_fn(batch):
    imgs, pids, camids, img_paths, mask_target, mask_target_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    mask_target = torch.tensor(mask_target, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, mask_target
    
def clustering_collate_fn(batch):
    imgs, pids, camids, img_paths, mask_target, mask_target_path = zip(*batch)
    return torch.stack(imgs, dim=0), mask_target_path, pids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths, mask_target, mask_target_path = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths
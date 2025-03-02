'''
Author: Li, Yirui
Date: 2021-08-07
Description: 
FilePath: /liyirui/PycharmProjects/ISP-reID/data/build.py
'''
# encoding: utf-8


from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn, clustering_collate_fn
from .datasets import init_dataset, ImageDataset, ImageDataset_train
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, pseudo_label_subdir=cfg.DATASETS.PSEUDO_LABEL_SUBDIR, part_num=cfg.CLUSTERING.PART_NUM, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, pseudo_label_subdir=cfg.DATASETS.PSEUDO_LABEL_SUBDIR, part_num=cfg.CLUSTERING.PART_NUM, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    # You can from train_set[i] get "img, pid, camid, img_path, align_target, align_target_path".
    # Note that align_target is pseudo_image, and align_target_path is pseudo_image_path
    # So, pseudo_image is containing in train set and you can get it from train_loader.
    train_set = ImageDataset_train(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    clustering_set = ImageDataset(dataset.train, val_transforms)
    clustering_loader = DataLoader(
        clustering_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=clustering_collate_fn
    )
    
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes, clustering_loader

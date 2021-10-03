'''
Author: Li, Yirui
Date: 2021-07-13
Description: 
FilePath: /liyirui/PycharmProjects/ISP-reID/layers/__init__.py
'''
# encoding: utf-8

from numpy.lib.function_base import angle
import torch
import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss
from .parsing_loss import CrossEntropy
import numpy as np

def make_loss(cfg, num_classes):    # modified by gu
    
    feat_dim = 256
    big_dim = 1920
    
    #parsing loss
    parsing_criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL)
    #triplet                                         
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    

    
    def loss_func(cls_score_part, cls_score_global, cls_score_fore, y_part, y_global, y_fore, part_pd_score, cls_target, part_target):
        
        loss = xent(cls_score_global, cls_target) + \
                xent(cls_score_fore, cls_target) + \
                xent(cls_score_part, cls_target) + \
                cfg.SOLVER.PARSING_LOSS_WEIGHT * parsing_criterion(part_pd_score, part_target) + \
                triplet(y_global, cls_target)[0] + \
                triplet(y_fore, cls_target)[0] + \
                triplet(y_part, cls_target)[0]
    
        return loss
                        
            
    return loss_func

    
def make_loss_with_center(cfg, num_classes):    # modified by gu
    
    feat_dim = 256
    big_dim = 1920
    
    #parsing loss
    parsing_criterion = CrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL)
    #triplet                                         
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)
    
    if cfg.MODEL.IF_WITH_CENTER == 'on':
        center_criterion_part = CenterLoss(num_classes=num_classes, feat_dim=feat_dim*(cfg.CLUSTERING.PART_NUM-1), use_gpu=True)
        if cfg.MODEL.IF_BIGG:
            center_criterion_global = CenterLoss(num_classes=num_classes, feat_dim=big_dim, use_gpu=True)
        else:
            center_criterion_global = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        center_criterion_fore = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    
    def loss_func(cls_score_part, cls_score_global, cls_score_fore, y_part, y_global, y_fore, part_pd_score, cls_target, part_target, angle_list = []):
        
        loss = xent(cls_score_global, cls_target) + \
                xent(cls_score_fore, cls_target) + \
                xent(cls_score_part, cls_target) + \
                cfg.SOLVER.PARSING_LOSS_WEIGHT * parsing_criterion(part_pd_score, part_target) + \
                triplet(y_global, cls_target)[0] + \
                triplet(y_fore, cls_target)[0] + \
                triplet(y_part, cls_target)[0] + \
                cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_global(y_global, cls_target) + \
                cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_fore(y_fore, cls_target) + \
                cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion_part(y_part, cls_target)
        
        # 判断是否加入角度信息
        # 如何在计算中使用角度信息做为约束？
        # 目前只是在全局特征和前景特征中加入了角度的距离约束
        if len(angle_list) > 0:
            if cfg.MODEL.IF_USE_ANGLE_MARGIN:
                g_angle_loss = margin_angle_loss(cfg, y_global, cls_target, angle_list)
                f_angle_loss = margin_angle_loss(cfg, y_fore, cls_target, angle_list)
                # p_angle_loss = margin_angle_loss(cfg, y_part, cls_target, angle_list)
            else:
                g_angle_loss = angle_loss(cfg, y_global, cls_target, angle_list)
                f_angle_loss = angle_loss(cfg, y_fore, cls_target, angle_list)
                # p_angle_loss = angle_loss(cfg, y_part, cls_target, angle_list)
            return loss + g_angle_loss + f_angle_loss 
        else:
            return loss
                        
            
    return loss_func, center_criterion_part, center_criterion_global, center_criterion_fore

def angle_loss(cfg, feat, cls_target, angle_list):
    same_id_diff_angle_loss = 0
    diff_id_same_angle_loss = 0
    for i in range(feat.shape[0]):
        for j in range(i+1, feat.shape[0]):
            if cls_target[i] == cls_target[j] and angle_list[i] != angle_list[j]:
                same_id_diff_angle_loss = same_id_diff_angle_loss + F.pairwise_distance(feat[i].unsqueeze(0), feat[j].unsqueeze(0), p=2)
            if cls_target[i] != cls_target[j] and angle_list[i] == angle_list[j]:
                distance = F.pairwise_distance(feat[i].unsqueeze(0), feat[j].unsqueeze(0), p=2)
                diff_id_same_angle_loss = diff_id_same_angle_loss + torch.exp(-1*distance)
    
    same_id_diff_angle_loss = cfg.MODEL.SAME_ID_DIFF_VIEW_WEIGHT * same_id_diff_angle_loss
    diff_id_same_angle_loss = cfg.MODEL.DIFF_ID_SAME_VIEW_WEIGHT * diff_id_same_angle_loss

    loss = 0
    if cfg.MODEL.SAME_ID_DIFF_VIEW:
        loss = loss + same_id_diff_angle_loss
    if cfg.MODEL.DIFF_ID_SAME_VIEW:
        loss = loss + diff_id_same_angle_loss

    return loss

def margin_angle_loss(cfg, feat, cls_target, angle_list):
    same_id_diff_angle_loss = 0
    diff_id_same_angle_loss = 0
    for i in range(feat.shape[0]):
        for j in range(i+1, feat.shape[0]):
            if cls_target[i] == cls_target[j] and angle_list[i] != angle_list[j]:
                distance = F.pairwise_distance(feat[i].unsqueeze(0), feat[j].unsqueeze(0), p=2)
                same_id_diff_angle_loss = same_id_diff_angle_loss + max(0, distance - cfg.MODEL.SAME_ID_DIFF_VIEW_MARGIN)
            if cls_target[i] != cls_target[j] and angle_list[i] == angle_list[j]:
                distance = F.pairwise_distance(feat[i].unsqueeze(0), feat[j].unsqueeze(0), p=2)
                diff_id_same_angle_loss = diff_id_same_angle_loss + max(0, cfg.MODEL.DIFF_ID_SAME_VIEW_MARGIN - distance)
    
    same_id_diff_angle_loss = cfg.MODEL.SAME_ID_DIFF_VIEW_WEIGHT * same_id_diff_angle_loss
    diff_id_same_angle_loss = cfg.MODEL.DIFF_ID_SAME_VIEW_WEIGHT * diff_id_same_angle_loss

    loss = 0
    if cfg.MODEL.SAME_ID_DIFF_VIEW:
        loss = loss + same_id_diff_angle_loss
    if cfg.MODEL.DIFF_ID_SAME_VIEW:
        loss = loss + diff_id_same_angle_loss

    return loss

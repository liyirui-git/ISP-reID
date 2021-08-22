# encoding: utf-8

import imp
import logging
from numpy.core.defchararray import array
import torch
import numpy
import os

from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_arm

img_name_dir = {}
img_name_list = []
feature_numpy = numpy.zeros((9462, 2048))

index = 0

def create_supervised_evaluator(cfg, model, metrics, 
                                device=None, with_arm=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)
    
    def _inference(engine, batch):
        '''
        batch is consist of:
            1. tensor format data
            2. pids
            3. camids
            4. img_paths
        
        Can get a reference from /ISP-reID/data/collate_batch.py 
        in "val_collate_fn()".
        '''
        global index

        model.eval()
        with torch.no_grad():
            # data: torch.Size([batch_size, 3, 256, 128])
            data, pids, camids, img_paths = batch
            # only take image_name
            if cfg.TEST.EXPORT_FEATURE:
                img_name = img_paths[0].split("/")[-1]
                if img_name not in img_name_dir: img_name_list.append(img_name)
            data = data.cuda()
            if with_arm:
                # g_f_feat.Size([batch_size, 512]) 这里是global_feat与foreground_feat连了起来
                # g_f_feat.Size([batch_size, 6, 256]) 这里是6个分开的part_feature
                g_f_feat, part_feat, part_visible, _ = model(data)
                return g_f_feat, part_feat, part_visible, pids, camids
            else:
                feat, _ = model(data)
                if cfg.TEST.EXPORT_FEATURE and img_name not in img_name_dir:
                    feat_cpu = feat.cpu().numpy()
                    for i in range(2048):
                        feature_numpy[index][i] = feat_cpu[0][i]
                    index = index + 1
                    img_name_dir[img_name] = 1
                return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query,
        output_dir
):

    device = cfg.MODEL.DEVICE
    with_arm = cfg.TEST.WITH_ARM
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")

    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        if cfg.TEST.EXPORT_FEATURE:
            assert cfg.TEST.IMS_PER_BATCH == 1, "cfg.TEST.IMS_PER_BATCH need set to 1"
        if with_arm:
            evaluator = create_supervised_evaluator \
            (cfg, model, metrics={'r1_mAP': R1_mAP_arm(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, 
            with_arm=with_arm)
        else:
            evaluator = create_supervised_evaluator \
            (cfg, model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
             device=device, 
             with_arm=with_arm)

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    
    # transfor list to numpy and restore
    if cfg.TEST.EXPORT_FEATURE and not cfg.TEST.WITH_ARM:
        # print("Number of feature is " + str(len(feature_list)))
        # print("Number of image path is " + str(len(img_name_list)))
        numpy.save(os.path.join(output_dir, "image_paths_of_features_new.npy"), array(img_name_list))
        numpy.save(os.path.join(output_dir, "features_new.npy"), feature_numpy)

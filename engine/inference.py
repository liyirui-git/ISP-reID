# encoding: utf-8

import imp
import logging
from numpy.core.defchararray import array
import torch
import numpy
import os
import cv2

from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_arm

img_part_pd_dir = {}
img_feat_dir = {}
feature_numpy = numpy.array([])
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
        global feature_numpy

        model.eval()
        with torch.no_grad():
            # data: torch.Size([batch_size, 3, 256, 128])
            data, pids, camids, img_paths = batch
            data = data.cuda()
            if with_arm:
                # g_f_feat.Size([batch_size, 512]) 这里是global_feat与foreground_feat连了起来
                # g_f_feat.Size([batch_size, 6, 256]) 这里是6个分开的part_feature
                g_f_feat, part_feat, part_visible, _, part_pd_score = model(data)
                return g_f_feat, part_feat, part_visible, pids, camids
            else:
                feat, _, part_pd_score = model(data)
                # part_pd_score ==> torch.Size([batch_size, 7, 64, 32])
                part_pd_score = part_pd_score.cpu()
                part_pd_score = part_pd_score[0].numpy()
                # only take image_name
                img_name = img_paths[0].split("/")[-1]
                if cfg.TEST.EXPORT_PART_PD_RESULT:
                    if img_name not in img_part_pd_dir: 
                        img_part_pd_dir[img_name] = part_pd_score
                if cfg.TEST.EXPORT_FEATURE:
                    if img_paths[0] not in img_feat_dir:
                        img_feat_dir[img_paths[0]] = 1
                        feat_cpu = feat.cpu()
                        if len(feature_numpy) == 0:
                            feature_numpy = feat_cpu
                        else:
                            feature_numpy = numpy.r_[feature_numpy, feat_cpu]
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

    # output query and gallery image features
    if cfg.TEST.EXPORT_FEATURE:
        logger.info("\nExport query and gallery feature begin: ")
        print("Number of feature is " + str(len(feature_numpy)))
        numpy.save(os.path.join(output_dir, "image_paths_of_features_new.npy"), array(list(img_feat_dir.keys())))
        numpy.save(os.path.join(output_dir, "features_new.npy"), feature_numpy)
        logger.info("Finish!\n")

    # output part prediction image 
    if cfg.TEST.EXPORT_PART_PD_RESULT:
        logger.info("\nExport part prediction of query and gallery begin: \n")
        # print 1400 images in query of BikePerson-700
        # max: 5.4795647
        # min: -6.5543547
        max_num, min_num = 4, -4
        part_pd_image_folder = os.path.join(cfg.OUTPUT_DIR, "part_prediction")
        if not os.path.exists(part_pd_image_folder):
            os.mkdir(part_pd_image_folder) 
        image_name_list = list(img_part_pd_dir.keys())
        img = numpy.zeros((64,32), numpy.uint8)
        # 使用白色填充图片区域,默认为黑色
        img.fill(255)
        for image_name in image_name_list:
            part_pd_score_images = img_part_pd_dir[image_name]
            # print(image_name)
            for i in range(cfg.CLUSTERING.PART_NUM):
                image = part_pd_score_images[i]
                for j in range(64):
                    for k in range(32):
                        if image[j][k] > max_num:
                            # print(image[j][k])
                            image[j][k] = max_num
                        if image[j][k] < min_num:
                            # trans
                            # print(image[j][k])
                            image[j][k] = max_num
                        img[j][k] = int((image[j][k] + max_num) * 255 / max_num / 2)
                im_color=cv2.applyColorMap(cv2.convertScaleAbs(img,alpha=1),cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(part_pd_image_folder, str(i)+"_"+image_name), im_color)
        logger.info("Finish!\n")

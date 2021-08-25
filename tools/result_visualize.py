'''
Author: your name
Date: 2021-08-25 09:06:14
LastEditTime: 2021-08-25 09:20:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /liyirui/PycharmProjects/ISP-reID/tools/result_visualize.py
'''
import os
import sys
from config import cfg
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
sys.path.append('.')
from utils.logger import setup_logger
from modeling import build_model

import numpy as np
import cv2

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(Cfg.LOG_DIR+ "/results/"):
        print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    cv2.imwrite(Cfg.LOG_DIR+ "/results/{}-cam{}.png".format(test_img,camid),figure)

if __name__ == "__main__":
    Cfg = cfg()
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark = True

    model = build_model(Cfg, 5000)
    model.load_param(Cfg.TEST.WEIGHT)

    device = 'cuda'
    model = model.to(device)
    transform = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)
    model.eval()
    for test_img in os.listdir(Cfg.QUERY_DIR):
        logger.info('Finding ID {} ...'.format(test_img))

        gallery_feats = torch.load(Cfg.LOG_DIR + 'feats.pth')
        img_path = np.load('./log/imgpath.npy')
        print(gallery_feats.shape, len(img_path))
        query_img = Image.open(Cfg.QUERY_DIR + test_img)
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            query_feat = model(input)

        dist_mat = cosine_similarity(query_feat, gallery_feats)
        indices = np.argsort(dist_mat, axis=1)
        visualizer(test_img, camid='mixed', top_k=10, img_size=Cfg.INPUT_SIZE)
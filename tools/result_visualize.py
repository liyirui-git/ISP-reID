'''
Author: your name
Date: 2021-08-25 09:06:14
LastEditTime: 2021-08-26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /liyirui/PycharmProjects/ISP-reID/tools/result_visualize.py
'''
import argparse
import imp
import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

sys.path.append('.')
from config import cfg
from utils.progress_bar import progress_bar


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def get_pid_camid(image_name):
    pid = int(image_name.split("_")[0])
    camid = int(image_name.split("_")[1][1:])
    return pid, camid

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

def visualizer(test_img, pid, camid, ap, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    empty = np.zeros((img_size[0], img_size[1], 3), np.uint8)
    figure2 = empty

    ct, getter = 0, 0
    while(True):
        i = indices[0][ct]
        ct = ct + 1
        #  剔除掉同一个摄像头，同一id的图片
        if pid == g_pids[i] and camid == g_camids[i]:
            continue
        img = np.asarray(Image.open(gallery_paths[indices[0][ct]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        getter = getter + 1
        if getter >= 10: break

    for i in range(len(gallery_paths)):
        if g_pids[i] == pid and g_camids[i] != q_camid:
            img = np.asarray(Image.open(gallery_paths[i]).resize((img_size[1],img_size[0])))
            figure2 = np.hstack((figure2, img))
    while(figure.shape != figure2.shape):
        figure2 = np.hstack((figure2, empty))
    # print(figure.shape)
    # print(figure2.shape)
    figure = np.vstack((figure, figure2))
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(LOG_DIR+ "/results/"):
        print('Creating a new folder named results in {}'.format(LOG_DIR))
        os.mkdir(LOG_DIR+ "/results/")
    cv2.imwrite(LOG_DIR+ "/results/{:.5f}-{}.png".format(round(ap, 5), test_img),figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    QUERY_DIR = os.path.join(cfg.DATASETS.DATASET_DIR, "query")
    DEVICE_ID = cfg.MODEL.DEVICE_ID
    INPUT_SIZE = cfg.INPUT.SIZE_TEST
    LOG_DIR = cfg.OUTPUT_DIR

    # os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
    # cudnn.benchmark = True

    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_feats = np.load(os.path.join(LOG_DIR, 'features.npy'))
    img_paths = np.load(os.path.join(LOG_DIR, "image_paths_of_features.npy"))

    query_img_dir = {}
    gallery_img_dir = {}
    image_path_index_dir = {}
    gallery_feats = np.array([])
    gallery_paths = []
    g_pids, g_camids = [], []

    print("\n------------------------ load features ------------------------")
    ct = 0
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img_name = img_path.split("/")[-1]
        pid, camid = get_pid_camid(img_name)
        if img_path.find("query") != -1:
            query_img_dir[img_name] = img_path
        else:
            gallery_img_dir[img_name] = img_path
            temp_feat = img_feats[ct]
            if len(gallery_feats) == 0:
                gallery_feats = temp_feat
            else: 
                gallery_feats = np.vstack((gallery_feats, temp_feat))
            gallery_paths.append(img_path)
            g_pids.append(pid)
            g_camids.append(camid)
        image_path_index_dir[img_path] = ct
        ct = ct + 1
        progress_bar(ct, len(img_paths))
        # if ct > 1500: break
    print("Finish!\n")

    print("------------------------ calculating AP and visualizing ------------------------")
    ct = 0
    ap_list = []
    gallery_feats = torch.from_numpy(gallery_feats)
    for query_img_name in os.listdir(QUERY_DIR):
        q_pid, q_camid = get_pid_camid(query_img_name)
        query_feat = torch.from_numpy(img_feats[image_path_index_dir[query_img_dir[query_img_name]]].reshape(1, -1))
        query_img = Image.open(os.path.join(QUERY_DIR, query_img_name))
        dist_mat = cosine_similarity(query_feat, gallery_feats)
        all_cmc, mAP = eval_func(dist_mat, np.array([q_pid]), np.array(g_pids), np.array([q_camid]), np.array(g_camids))
        # because this is only one image in query, so mAP is AP
        ap_list.append(mAP)
        # print("AP: ", mAP)
        indices = np.argsort(dist_mat, axis=1)
        visualizer(query_img_name, pid=q_pid ,camid=q_camid, ap=mAP, top_k=10, img_size=INPUT_SIZE)
        ct = ct + 1
        progress_bar(ct, len(query_img_dir))
    print("Finish!\n")

    print("------------------------ calculating mAP ------------------------")
    total_ap = 0
    for ap in ap_list:
        total_ap = total_ap + ap
    mAP = total_ap / len(ap_list)
    print('mAP: ', mAP)
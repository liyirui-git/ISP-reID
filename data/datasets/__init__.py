'''
Author: your name
Date: 2021-07-13 11:47:58
LastEditTime: 2021-11-16 09:02:12
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /liyirui/PycharmProjects/ISP-reID/data/datasets/__init__.py
'''
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .dukemtmcreid import DukeMTMCreID
from .occluded_dukemtmcreid import OccludedDukeMTMCreID
from .market1501 import Market1501
from .dataset_loader import ImageDataset, ImageDataset_train
from .cuhk03_np_labeled import CUHK03_NP_labeled
from .cuhk03_np_detected import CUHK03_NP_detected
from .bikeperson import BikePerson
from .bikeperson_verification import BikePerson_Verification
from .dml_ped import DML_PED


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'occluded_dukemtmc': OccludedDukeMTMCreID,
    'cuhk03_np_labeled': CUHK03_NP_labeled,
    'cuhk03_np_detected': CUHK03_NP_detected,
    'bikeperson': BikePerson,
    'bikeperson_ver': BikePerson_Verification,
    'dml_ped': DML_PED
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

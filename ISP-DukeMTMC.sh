python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('0')" MODEL.IF_WITH_CENTER "('no')" MODEL.NAME "('HRNet32')" MODEL.PRETRAIN_PATH "('./hrnetv2_w32_imagenet_pretrained.pth')" DATASETS.NAMES "('dukemtmc')" DATASETS.ROOT_DIR "('/home/liyirui/PycharmProjects/dataset/')" CLUSTERING.PART_NUM "(7)" DATASETS.PSEUDO_LABEL_SUBDIR "('train_pseudo_labels-ISP-7')"  OUTPUT_DIR "('./log/ISP-Duke-7')"

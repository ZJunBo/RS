"""
# -*- coding:utf-8 -*-
@Project : DCA-master
@File : generate_class_distribution.py.py
@Author : ZhangJunBo
@Time : 2023/3/16 上午9:08
"""
import pdb
import time

from data.loveda import LoveDALoader
import logging
import numpy as np
logger = logging.getLogger(__name__)
from utils.tools import *
from utils.my_tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm


def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    save_path = '/home/zjb/PycharmProjects/DCA-master/Pseudo'
    pred_cls_num = np.zeros(num_classes)
    true_cls_num = np.zeros(num_classes)
    sm = nn.Softmax(dim=0)
    model.eval()
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader): # ret [1,3,1024,1024]
            ret = ret.to(torch.device('cuda'))
            preds1= model(ret) # [1,7,1024,1024]
            # feat = feats  # [1,256,129,257]
            # bs, _, h, w = feat.shape

            # for i in range(preds1.shape[0]):  # [1,19,129,257]
            #     output_cb = sm(preds1[i]).cpu().numpy().transpose(1, 2, 0)
            #     amax_output = np.asarray(np.argmax(output_cb, axis=2), dtype=np.uint8)
            #     conf = np.amax(output_cb, axis=2)
            #     pred_label = amax_output.copy()
            #     for idx_cls in range(num_classes):
            #         idx_temp = pred_label == idx_cls
            #         pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
            output_cb = sm(preds1[0]).cpu().numpy().transpose(1, 2, 0)
            amax_output = np.asarray(np.argmax(output_cb, axis=2), dtype=np.uint8)
            pred_label = amax_output.copy()

            cls_gt = ret_gt['cls'][0].cpu().numpy()
            true_label = cls_gt.copy()
            for idx_cls in range(num_classes):
                idx_temp = pred_label == idx_cls
                pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)

                true_temp = true_label==idx_cls
                true_cls_num[idx_cls] = true_cls_num[idx_cls] + np.sum(true_temp)
        class_distribution = pred_cls_num / np.sum(pred_cls_num)
        true_class_distribution = true_cls_num / np.sum(true_cls_num)
        print(class_distribution)
        print(true_class_distribution)
        np.save(os.path.join(save_path, "class_distribution_rural.npy"), class_distribution)


    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(2333)
    ckpt_path = '/home/zjb/PycharmProjects/uda/log/ast/2rural/RURAL8000.pth'
    from module.Encoder import Deeplabv2

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=False,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    )).cuda()
    num_classes = 7
    cfg = import_config('ast.2rural')
    logger = get_console_file_logger(name='ast', logdir=cfg.SNAPSHOT_DIR)
    evaluate(model, cfg, False, ckpt_path, logger)
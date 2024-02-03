"""
# -*- coding:utf-8 -*-
@Project : DCA-master
@File : calc_prototype.py.py
@Author : ZhangJunBo
@Time : 2023/3/15 下午9:42
"""
import time
import pdb
from data.loveda import LoveDALoader
import logging
import numpy as np
logger = logging.getLogger(__name__)
from utils.tools import *
from utils.my_tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm

class Class_Features:
    def __init__(self, numbers = 7):
        self.class_numbers = numbers
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(torch.device('cuda'))
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(torch.device('cuda')))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None, model=None):
        # pdb.set_trace()
        outputs_softmax = F.softmax(outputs, dim=1) # torch.Size([1, 7, 1024, 1024])
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True) # torch.Size([1, 1, 1024, 1024])
        outputs_argmax = self.process_label(label = outputs_argmax.float()) # torch.Size([1, 8, 1024, 1024])
        if labels_val is None:
            outputs_pred = outputs_argmax #  torch.Size([1, 8, 1024, 1024])
        else:
            labels_expanded = model.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1) # torch.Size([1, 8, 1, 1])
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]): # feat_cls.size() torch.Size([1, 7, 1024, 1024])
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

def evaluate(model, cfg, is_training=True, ckpt_path=None, logger=None):
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict, strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)

    save_path = '/home/zjb/PycharmProjects/uda'
    save_path = os.path.join(save_path,"prototypes_on_rural_to_urban")
    class_features = Class_Features(numbers=7)
    model.eval()
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader): # ret [1,3,1024,1024]
            ret = ret.to(torch.device('cuda'))
            preds1, feats = model(ret) # orch.Size([1, 7, 64, 64]); torch.Size([1, 256, 64, 64])
            vectors, ids = class_features.calculate_mean_vector(feats, preds1, model=model)
            for t in range(len(ids)):
                model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu(), 'mean')

    torch.save(model.objective_vectors, save_path)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    seed_torch(2333)
    ckpt_path = '/home/zjb/PycharmProjects/uda/log/ast/urban/URBAN10000.pth'
    from module.Encoder import Deeplabv2Test

    model = Deeplabv2Test(dict(
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

    cfg = import_config('ast.2urban')
    logger = get_console_file_logger(name='adv_self', logdir=cfg.SNAPSHOT_DIR)
    evaluate(model, cfg, False, ckpt_path, logger)
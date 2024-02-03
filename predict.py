import pdb

from data.loveda import LoveDALoader, TestLoader
from utils.tools import *
from utils.my_tools import *
from module.viz import VisualizeSegmm
from skimage.io import imsave
import os
import torch.nn as nn
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def predict_test(model, cfg, ckpt_path=None, save_dir='./submit_test/ast/urban'):
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(2333)
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict,  strict=True)
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir+'_TEST', palette)



    # count_model_parameters(model)
    model.eval()
    print(cfg.TEST_DATA_CONFIG)
    test_dataloader = TestLoader(cfg.TEST_DATA_CONFIG)

    with torch.no_grad():
        for ret in tqdm(test_dataloader):
            rgb = ret['rgb'].to(torch.device('cuda'))
            cls = model(rgb)
            # print(cls.shape)
            # cls = pre_slide(model, rgb, num_classes=7, tile_size=(512, 512), tta=True)

            cls = cls.argmax(dim=1).cpu().numpy()

            cv2.imwrite(save_dir + '/' + ret['fname'][0], cls.reshape(1024, 1024).astype(np.uint8))
            for fname, pred in zip(ret['fname'], cls):
                viz_op(pred, fname.replace('tif', 'png'))


    torch.cuda.empty_cache()


if __name__ == '__main__':
    ckpt_path = '/home/zjb/PycharmProjects/uda/log/ast/urban/URBAN10000.pth'
    from module.Encoder import Deeplabv2
    cfg = import_config('ast.2urban')
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
    predict_test(model, cfg, ckpt_path)
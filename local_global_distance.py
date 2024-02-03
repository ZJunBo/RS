import pdb

from data.loveda import LoveDALoader
import logging
logger = logging.getLogger(__name__)
from utils.tools import *
from ever.util.param_util import count_model_parameters
from module.viz import VisualizeSegmm
from ever.core.iterator import Iterator
from module.my_modules import *



def evaluate(model, cfg, is_training=False, ckpt_path=None, logger=None):
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
    if cfg.SNAPSHOT_DIR is not None:
        vis_dir = os.path.join(cfg.SNAPSHOT_DIR, 'vis-{}'.format(os.path.basename(ckpt_path)))
        palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
        viz_op = VisualizeSegmm(vis_dir, palette)
    if not is_training:
        model_state_dict = torch.load(ckpt_path)
        model.load_state_dict(model_state_dict,  strict=True)
        logger.info('[Load params] from {}'.format(ckpt_path))
        count_model_parameters(model, logger)
    model.eval()
    # print(cfg.EVAL_DATA_CONFIG)
    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    eval_dataloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    evaldataloader_iter = Iterator(eval_dataloader)

    i_iter = 0
    iterations = 500
    Score_local = np.zeros((7,))
    Score_global = 0
    while i_iter < iterations:
        # source output
        batch_s = trainloader_iter.next()
        images_s, label_s = batch_s[0]
        images_s, lab_s = images_s.cuda(), label_s['cls'].cuda()
        # target output
        batch_t = evaldataloader_iter.next()
        images_t, label_t = batch_t[0]
        images_t, lab_t = images_t.float().cuda(), label_t['cls'].cuda()  # labe_t shape torch.Size([4, 7, 512, 512])
        # model forward
        # source
        pred_s,  feat_s = model(images_s) # [4. 7, 32, 32]  [4, 256, 32, 32]
        # target
        pred_t, feat_t = model(images_t)  # [4. 7, 32, 32]  [4, 256, 32, 32]

        class_distance = Class_Distance_Compute([pred_s, feat_s], [pred_t, feat_t])
        Score_local += class_distance

        global_distance = Global_Distance_Compute(pred_s, pred_t)
        Score_global += global_distance
        i_iter += 1
        # print("当前迭代次数为{}".format(i_iter))
        # print(class_distance)
        # pdb.set_trace()
    Score_local_class = Score_local/iterations
    Score_global = Score_global/iterations
    Score_local_mean = np.sum(Score_local_class)/7
    uniform_score = Score_global * Score_local_mean
    print("局部对齐类别分数为{}".format(Score_local_class))
    print("局部对齐分数为{}".format(Score_local_mean))
    print("全部对齐分数为{}".format(Score_global))
    print("综合对齐分数为{}".format(uniform_score))
    print("计算结束")

    torch.cuda.empty_cache()



if __name__ == '__main__':
    from module.Encoder import Deeplabv2Test

    seed_torch(2333)
    ckpt_path = '/home/zjb/PycharmProjects/uda/log/ast/2rural/RURAL6000.pth'
    from module.Encoder import Deeplabv2
    cfg = import_config('st.cbst.2rural')
    logger = get_console_file_logger(name='Baseline', logdir=cfg.SNAPSHOT_DIR)
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
    evaluate(model, cfg, False, ckpt_path, logger)


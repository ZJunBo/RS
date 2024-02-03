import argparse
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.optim as optim
import os.path as osp
# from module.CLAN_G import Deeplabv2
from module.Encoder import Deeplabv2
from module.Discriminator import FCDiscriminator, PixelDiscriminator
from data.loveda import LoveDALoader, LoveDALoader2
from ever.core.iterator import Iterator
from utils.tools import *
from tqdm import tqdm
from eval import evaluate
from torch.nn.utils import clip_grad
from module.viz import VisualizeSegmm
import torch.backends.cudnn as cudnn
import warnings
import torch.nn.functional
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='Run My Methods.')

parser.add_argument('--config_path',  type=str, default='ast.2rural', help='config path')
args = parser.parse_args()
cfg = import_config(args.config_path)

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def main():
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='adversarial and self-training', logdir=cfg.SNAPSHOT_DIR)
    cudnn.enabled = True
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_soft_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'soft_pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.


    if not os.path.exists(cfg.SNAPSHOT_DIR):
        os.makedirs(cfg.SNAPSHOT_DIR)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_soft_pseudo_label_path):
        os.makedirs(save_soft_pseudo_label_path)

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
    ))
    model.train()
    model.cuda()
    logger.info('exp = %s' % cfg.SNAPSHOT_DIR)
    # init D
    model_D1 = FCDiscriminator(cfg.NUM_CLASSES)
    model_D2 = FCDiscriminator(cfg.NUM_CLASSES)
    model_D_category = PixelDiscriminator(2048, num_classes=7)

    model_D1.train()
    model_D1.cuda()

    model_D2.train()
    model_D2.cuda()

    model_D_category.train()
    model_D_category.cuda()

    count_model_parameters(model, logger)

    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    trainloader_iter = Iterator(trainloader)
    targetloader = LoveDALoader(cfg.TARGET_DATA_CONFIG)
    targetloader_iter = Iterator(targetloader)
    evalloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)

    epochs = cfg.NUM_STEPS_MAX / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)

    optimizer = optim.SGD(model.parameters(), lr=cfg.AT_LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.AT_WEIGHT_DECAY)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=cfg.AT_LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=cfg.AT_LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    optimizer_D = torch.optim.Adam(model_D_category.parameters(), lr=cfg.AT_LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    source_label = 0
    target_label = 1

    for i_iter in tqdm(range(cfg.NUM_STEPS_MAX)):
        #---------------------------adversarial training stage----------------------------------------
        if i_iter <= cfg.AT_traing_steps:
            loss_seg_value1 = 0
            loss_adv_target_value1 = 0
            loss_D_value1 = 0

            loss_seg_value2 = 0
            loss_adv_target_value2 = 0
            loss_D_value2 = 0

            loss_adv_target_category = 0
            loss_D_category = 0

            optimizer.zero_grad()
            G_lr = adv_adjust_learning_rate(optimizer, i_iter, cfg)

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_D.zero_grad()
            D_lr = adv_adjust_learning_rate_D(optimizer_D1, i_iter, cfg)
            adv_adjust_learning_rate_D(optimizer_D2, i_iter, cfg)


            for sub_i in range(cfg.AT_ITER_SIZE):
                # train G
                # don't accumulate grads in D
                for param in model_D1.parameters():
                    param.requires_grad = False

                for param in model_D2.parameters():
                    param.requires_grad = False

                for param in model_D_category.parameters():
                    param.requires_grad = False
                # train with source
                temperature = 1.8
                batch = trainloader_iter.next()
                images, labels = batch[0]
                b1, c1, h1, w1 = images.shape
                images = Variable(images).cuda()
                pred1_feat_out, pred2_feat_out, feat_s = model(images)  # torch.Size([4, 7, 32, 32]),torch.Size([4, 7, 32, 32]), torch.Size([4, 2048, 32, 32])
                pred1, pred2 = pred1_feat_out['out'], pred2_feat_out['out']
                pred1 = F.interpolate(pred1, size=labels['cls'].size()[-2:], mode='bilinear', align_corners=True)
                pred2 = F.interpolate(pred2, size=labels['cls'].size()[-2:], mode='bilinear', align_corners=True)
                source_pred = (pred1+pred2)/2
                source_pred = source_pred.div(temperature)
                loss_seg1 = loss_calc(pred1, labels['cls'].cuda(), multi=False)  # labels['cls'] [8,512,512]
                loss_seg2 = loss_calc(pred2, labels['cls'].cuda(), multi=False)
                loss = loss_seg2 + cfg.AT_LAMBDA_SEG * loss_seg1

                # proper normalization
                loss = loss / cfg.AT_ITER_SIZE
                loss.backward()
                loss_seg_value1 += loss_seg1.data.cpu().numpy() / cfg.AT_ITER_SIZE
                loss_seg_value2 += loss_seg2.data.cpu().numpy() / cfg.AT_ITER_SIZE

                # train with target
                batch = targetloader_iter.next()
                images, labels = batch[0]
                images = Variable(images).cuda()
                pred1_target_feat_out, pred2_target_feat_out, feat_t = model(images)
                pred_target1, pred_target2 = pred1_target_feat_out['out'], pred2_target_feat_out['out']
                pred_target1 = F.interpolate(pred_target1, size=labels['cls'].size()[-2:], mode='bilinear', align_corners=True)
                pred_target2 = F.interpolate(pred_target2, size=labels['cls'].size()[-2:], mode='bilinear', align_corners=True)
                tar_pred = (pred_target1+pred_target2)/2
                tar_pred = tar_pred.div(temperature)

                # generate soft labels
                src_soft_label = F.softmax(source_pred, dim=1).detach()
                src_soft_label[src_soft_label > 0.9] = 0.9
                tar_soft_label = F.softmax(tar_pred, dim=1).detach()
                tar_soft_label[tar_soft_label > 0.9] = 0.9

                D_out1 = model_D1(F.softmax(pred_target1))
                D_out2 = model_D2(F.softmax(pred_target2))
                tar_D_pred = model_D_category(feat_t)
                tar_D_pred = F.interpolate(tar_D_pred, size=(h1, w1), mode='bilinear', align_corners=True)

                loss_adv_target1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
                loss_adv_target2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
                loss_adv_tgt = 0.001 * soft_label_cross_entropy(tar_D_pred, torch.cat((tar_soft_label, torch.zeros_like(tar_soft_label)), dim=1))

                loss = 0.001  * loss_adv_target1 + 0.0002 * loss_adv_target2 + loss_adv_tgt
                loss = loss / cfg.AT_ITER_SIZE
                loss.backward()
                loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / cfg.AT_ITER_SIZE
                loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / cfg.AT_ITER_SIZE
                loss_adv_target_category += loss_adv_tgt.data.cpu().numpy() / cfg.AT_ITER_SIZE

                # train D

                # bring back requires_grad
                for param in model_D1.parameters():
                    param.requires_grad = True
                for param in model_D2.parameters():
                    param.requires_grad = True
                for param in model_D_category.parameters():
                    param.requires_grad = True

                # train with source
                pred1 = pred1.detach()
                pred2 = pred2.detach()
                feat_s = feat_s.detach()

                D_out1 = model_D1(F.softmax(pred1))
                D_out2 = model_D2(F.softmax(pred2))
                D_out3 = model_D_category(feat_s)
                src_D_pred = F.interpolate(D_out3, size=(h1, w1), mode='bilinear', align_corners=True)

                loss_D1 = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda())
                loss_D2 = bce_loss(D_out2, Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda())
                loss_D_src = 0.5 * soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1))

                loss_D1 = loss_D1 / cfg.AT_ITER_SIZE / 2
                loss_D2 = loss_D2 / cfg.AT_ITER_SIZE / 2
                loss_D_src = loss_D_src / cfg.AT_ITER_SIZE / 2

                loss_D1.backward()
                loss_D2.backward()
                loss_D_src.backward()


                loss_D_value1 += loss_D1.data.cpu().numpy()
                loss_D_value2 += loss_D2.data.cpu().numpy()
                loss_D_category += loss_D_src.data.cpu().numpy()

                # train with target
                pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()
                feat_t = feat_t.detach()


                D_out1 = model_D1(F.softmax(pred_target1))
                D_out2 = model_D2(F.softmax(pred_target2))
                tgt_D_pred = model_D_category(feat_t)
                tgt_D_pred = F.interpolate(tgt_D_pred, size=(h1, w1), mode='bilinear', align_corners=True)


                loss_D1 = bce_loss(D_out1,  Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda())
                loss_D2 = bce_loss(D_out2,  Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda())
                loss_D_tgt = 0.5 * soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tar_soft_label), tar_soft_label), dim=1))

                loss_D1 = loss_D1 / cfg.AT_ITER_SIZE / 2
                loss_D2 = loss_D2 / cfg.AT_ITER_SIZE / 2
                loss_D_tgt = loss_D_tgt / cfg.AT_ITER_SIZE / 2

                loss_D1.backward()
                loss_D2.backward()
                loss_D_tgt.backward()

                loss_D_value1 += loss_D1.data.cpu().numpy()
                loss_D_value2 += loss_D2.data.cpu().numpy()
                loss_D_category += loss_D_tgt.data.cpu().numpy()

            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D1.parameters()), max_norm=35, norm_type=2)
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D2.parameters()), max_norm=35, norm_type=2)
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D_category.parameters()), max_norm=35, norm_type=2)

            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_D.step()


            if i_iter % 50 == 0:
                logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                logger.info(
                    'iter = %d loss_seg1 = %.3f loss_seg2 = %.3f loss_adv1 = %.3f, loss_adv2 = %.3f loss_D1 = %.3f loss_D2 = %.3f loss_D3 = %.3f G_lr = %.5f D_lr = %.5f' % (
                        i_iter, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2, loss_D_category, G_lr, D_lr)
                )

            if i_iter % cfg.AT_EVAL_EVERY == 0 and i_iter != 0:
                print('taking snapshot ...')
                ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                torch.save(model.state_dict(), ckpt_path)
                evaluate(model, cfg, True, ckpt_path, logger)
                model.train()


        else:
            #----------------------------self-training stage---------------------------------------
            # Generate pseudo label
            if (i_iter-1) % cfg.GENERATE_PSEDO_EVERY == 0 or targetloader is None:
                # 加载类原型
                print("开始加载类别原型")
                category_anchors_path = "/home/zjb/PycharmProjects/uda/prototypes_on_rural_to_urban.pth"
                objective_vectors = torch.load(category_anchors_path)
                model.objective_vectors = torch.Tensor(objective_vectors).cuda()
                class_distribution_path = "/home/zjb/PycharmProjects/uda/class_distribution.npy"
                class_distribution = np.load(class_distribution_path)
                model.class_distribution = torch.Tensor(class_distribution).cuda()
                # init prototype layer
                w = torch.nn.functional.normalize(model.objective_vectors, dim=1, p=2)  # shape: [7, 256]
                model.prototypes.data = w

                logger.info('###### Start generate pseudo dataset in round {}! ######'.format(i_iter))
                # save soft pseudo label for target domain
                gener_target_soft_pseudo(cfg, model, evalloader, save_soft_pseudo_label_path)

                # save finish
                target_config = cfg.TARGET_DATA_CONFIG
                target_config['mask_dir'] = [save_soft_pseudo_label_path]
                logger.info(target_config)
                targetloader = LoveDALoader2(target_config)
                targetloader_iter = Iterator(targetloader)
            if i_iter == (cfg.AT_traing_steps + 1):
                logger.info('###### Start the self-training Stage in round {}! ######'.format(i_iter))

            torch.cuda.synchronize()
            #beigin self-traing
            if i_iter < cfg.NUM_STEPS_MAX and targetloader is not None:
                model.train()
                lr = adjust_learning_rate(optimizer, i_iter, cfg)

                # source output
                batch_s = trainloader_iter.next()
                images_s, label_s = batch_s[0]
                images_s, lab_s = images_s.cuda(), label_s['cls'].cuda()
                # target output
                batch_t = targetloader_iter.next()
                images_t, label_t = batch_t[0]
                images_t, lab_t = images_t.float().cuda(), label_t['cls'].cuda() # labe_t shape torch.Size([4, 7, 512, 512])
                lab_t_hard = lab_t.argmax(dim=1)
                # model forward
                # source
                pred_s1_feat_out, pred_s2_feat_out, feat_s = model(images_s)
                pred_s1, pred_s2 = pred_s1_feat_out['out'], pred_s2_feat_out['out']
                # target
                pred_t1_feat_out, pred_t2_feat_out, feat_t = model(images_t)  # torch.Size([4, 7, 32, 32]), torch.Size([4, 7, 32, 32]), torch.Size([4, 2048, 32, 32])
                pred_t1, pred_t2 = pred_t1_feat_out['out'], pred_t2_feat_out['out']
                loss_hard_pseudo = loss_calc([pred_t1, pred_t2], lab_t_hard, multi=True)

                feat_SL = (pred_t1_feat_out['feat']+ pred_t2_feat_out['feat'])/2 # [4,256,32,32]
                # ============ recitify pseudo label ============
                feat_target = feat_SL # [4,256,32,32]
                weights = model.get_prototype_weight(feat_target)  # torch.Size([4, 7, 32, 32])
                lab_t =  F.interpolate(lab_t, size = (32,32), mode='bilinear', align_corners=True)  # 做的下采样[4,7,512,512] -> [4, 7, 32, 32]
                rectified = weights * lab_t # 【4，7，512，512】
                threshold_arg = rectified.max(1, keepdim=True)[1] # 维度不匹配 [4,1,512,512]
                rectified = rectified / rectified.sum(1, keepdim=True) # [4,7,512,512]
                argmax = rectified.max(1, keepdim=True)[0] # [4,1,512,512]
                threshold_arg[argmax < 0] = -1
                bs, channle, h, w = threshold_arg.shape
                loss_soft_pseudo = loss_calc((pred_t1+pred_t2)/2, threshold_arg.reshape(bs, h, w), multi=False)  # L_t_seg [4,7,32,32] 与[4,32,32]做loss计算
                loss_pseudo = loss_hard_pseudo + 0.01*loss_soft_pseudo
                # ============ self-labeling loss ============
                feat_SL = feat_SL.transpose(1, 2).transpose(2, 3).contiguous().view(bs, -1, 256)  # [bs, h*w, 256] torch.Size([4, 1024, 256])
                # randomly sampling pixel features
                rand_index = torch.randperm(feat_SL.shape[1])  # 将0~8192随机打乱后获得的数字序列
                feat_SL = feat_SL[:, rand_index]  # [4,8192,256]，涉及到多维数组的切片
                feat_SL_DS = feat_SL[:, :256]  # [4,256,256]，取的一个feature batch
                feat_SL_DS = torch.nn.functional.normalize(feat_SL_DS, dim=2, p=2)  # [4,256,256]

                # use feat*proto to produce Q
                loss_SL = 0
                for i in range(feat_SL_DS.shape[0]):  # 从每一个batch取
                    proto = torch.nn.functional.normalize(model.prototypes, dim=1, p=2)  # self.prototypes [7,256]
                    out = torch.mm(feat_SL_DS[i], proto.t())  # [256,256] x [256,7] =[256,7]
                    with torch.no_grad():
                        out_ = out.detach()
                        # get assignments
                        q = model.sinkhorn(out_)[-256:]  # [256,7]
                    p = out / 0.08  # [256,7] / 0.08
                    loss_SL -= torch.mean(torch.sum(q * F.log_softmax(p, dim=1), dim=1))
                loss_SL /= feat_SL_DS.shape[0]

                # loss
                loss_seg = loss_calc([pred_s1, pred_s2], lab_s, multi=True)

                LAMBDA_SL = 0.001
                loss = loss_seg + loss_pseudo  + LAMBDA_SL*loss_SL

                optimizer.zero_grad()
                loss.backward()
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
                optimizer.step()

                if i_iter % 50 == 0:
                    logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
                    text = 'iter = %d, total = %.3f, seg = %.3f, pseudo = %.3f, sl = %.3f,  lr = %.3f' % (
                    i_iter, loss, loss_seg, loss_pseudo, loss_SL,lr)
                    logger.info(text)

                if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
                    ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    evaluate(model, cfg, True, ckpt_path, logger)
                    model.train()


def gener_target_pseudo(cfg, model, evalloader, save_pseudo_label_path):
    model.eval()

    save_pseudo_color_path = save_pseudo_label_path + '_color'
    if not os.path.exists(save_pseudo_color_path):
        os.makedirs(save_pseudo_color_path)
    viz_op = VisualizeSegmm(save_pseudo_color_path, palette)

    with torch.no_grad():
        for ret, ret_gt in tqdm(evalloader):
            ret = ret.to(torch.device('cuda'))

            cls = model(ret)
            # pseudo selection, from -1~6
            if cfg.PSEUDO_SELECT:
                cls = pseudo_selection(cls)
            else:
                cls = cls.argmax(dim=1).cpu().numpy()

            cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
                        (cls + 1).reshape(1024, 1024).astype(np.uint8))

            if cfg.SNAPSHOT_DIR is not None:
                for fname, pred in zip(ret_gt['fname'], cls):
                    viz_op(pred, fname.replace('tif', 'png'))

def gener_target_soft_pseudo(cfg, model, evalloader, save_soft_pseudo_label_path):
    model.eval()
    with torch.no_grad():
        for ret, ret_gt in tqdm(evalloader):
            ret = ret.to(torch.device('cuda'))
            cls = model(ret)
            soft_pseudo = F.softmax(cls, dim=1) # [bs,7,1024,1024]
            for k in range(cls.shape[0]):
                name = ret_gt['fname'][0].replace('.png', '.npy')
                np.save(save_soft_pseudo_label_path + '/' + name, soft_pseudo[k].cpu().numpy())


def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6):
    """Convert continuous mask into binary mask"""
    assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)
    mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)
    # remove ambiguous pixels, ambiguous = 1 means ignore
    ambiguous = (pseudo_gt.sum(1, keepdim=True) != 1).type_as(mask)

    pseudo_gt = pseudo_gt.argmax(dim=1, keepdim=True)
    pseudo_gt[ambiguous == 1] = -1

    return pseudo_gt.view(bs, h, w).cpu().numpy()




if __name__ == '__main__':
    seed_torch(2333)
    main()

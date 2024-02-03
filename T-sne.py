import pdb
import time

import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
import torch.nn as nn
from data.loveda import LoveDALoader
from utils.tools import *
from tqdm import tqdm
import torch.nn.functional as F

class RunTsne():
    def __init__(self,
                selected_cls,        # 选择可视化几个类别
                domId2name,          # 不同域的ID
                trainId2name,        # 标签中每个ID所对应的类别
                trainId2color=None,  # 标签中每个ID所对应的颜色
                output_dir='./',     # 保存的路径
                tsnecuda=True,       # 是否使用tsnecuda，如果不使用tsnecuda就使用MulticoreTSNE
                extention='.png',    # 保存图片的格式
                duplication=10):     # 程序循环运行几次，即保存多少张结果图片
        self.tsne_path = output_dir
        os.makedirs(self.tsne_path, exist_ok=True)
        self.domId2name = domId2name # {0: 'gtav', 1: 'synthia', 2: 'cityscapes', 3: 'bdd100k', 4: 'mapillary', 5: 'idd'}
        self.name2domId = {v:k for k,v in domId2name.items()} # {'gtav': 0, 'synthia': 1, 'cityscapes': 2, 'bdd100k': 3, 'mapillary': 4, 'idd': 5}
        self.trainId2name = trainId2name
        self.trainId2color = trainId2color
        self.selected_cls = selected_cls # ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light']
        self.name2trainId = {v:k for k,v in trainId2name.items()}
        #{'trailer': 255, 'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 'fence': 4, 'pole': 5, 'traffic light': 6, 'traffic sign': 7, 'vegetation': 8, 'terrain': 9, 'sky': 10, 'person': 11, 'rider': 12, 'car': 13, 'truck': 14, 'bus': 15, 'train': 16, 'motorcycle': 17, 'bicycle': 18, 'license plate': -1}
        # self.selected_clsid = [self.name2trainId[x] for x in selected_cls]
        self.selected_clsid = [0, 1, 2, 3, 4, 5, 6]
        self.tsnecuda = False
        self.extention = extention
        self.num_class = 7
        self.duplication = duplication

        self.init_basket()    # 初始化

        if self.tsnecuda:
            from tsnecuda import TSNE
            self.max_pointnum = 9000    # 最大特征向量的数量
            self.perplexity = 30        # 未知
            self.learning_rate = 100    # t-SNE的学习率
            self.n_iter = 3500          # t-SNE迭代步数
            self.num_neighbors = 128    # 未知，以上几个参数是针对t-SNE比较重要的参数，可以根据自己的需要进行调整
            self.TSNE = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate, metric='innerproduct',
                 random_seed=304, num_neighbors=self.num_neighbors, n_iter=self.n_iter, verbose=1)
        else:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            self.max_pointnum = 10200
            self.perplexity = 50
            self.learning_rate = 4800
            self.n_iter = 3000
            self.TSNE = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                             n_iter=self.n_iter, verbose=1, n_jobs=4)

    def init_basket(self):
        self.feat_vecs = torch.tensor([]).cuda()            # 特征向量
        self.feat_vec_labels = torch.tensor([]).cuda()      # 特征向量的类别
        self.feat_vec_domlabels = torch.tensor([]).cuda()   # 特征向量的域信息
        self.mem_vecs = None                                # 聚类中心的向量
        self.mem_vec_labels = None                          # 聚类中心的类别

    def input_memory_item(self,m_items):
        self.mem_vecs = m_items[self.selected_clsid]
        self.mem_vec_labels = torch.tensor(self.selected_clsid).unsqueeze(dim=1).squeeze()

    def input2basket(self, feature_map, gt_cuda, datasetname):
        b, c, h, w = feature_map.shape
        features = F.normalize(feature_map.clone(), dim=1)
        gt_cuda = gt_cuda.clone()
        H, W = gt_cuda.size()[-2:]
        gt_cuda[gt_cuda == -1] = self.num_class
        gt_cuda = F.one_hot(gt_cuda, num_classes=self.num_class + 1)

        gt = gt_cuda.view(1, -1, self.num_class + 1) #[1,HxW，c+1],torch.Size([1, 65536, 8])
        denominator = gt.sum(1).unsqueeze(dim=1) # 空间维度求和并挤压掉这个维度
        denominator = denominator.sum(0)  # batchwise sum
        denominator = denominator.squeeze() # [8,0],tensor([22620, 17409,     0,     0,     0,  2928, 22579, 0],   device='cuda:0')

        features = F.interpolate(features, [H, W], mode='bilinear', align_corners=True) # [1,2048,1024,1024]
        # 这里是将feature采样到跟标签一样的大小。当然也可以将标签采样到跟feature一样的大小
        features = features.view(b, c, -1) # torch.Size([1, 2048, 1048576])
        nominator = torch.matmul(features, gt.type(torch.float32)) # [1,2048,8]
        nominator = torch.t(nominator.sum(0))  # batchwise sum torch.Size([8, 2048])
        for slot in self.selected_clsid:
            if denominator[slot] != 0:
                cls_vec = nominator[slot] / denominator[slot]  # mean vector torch.Size([7])
                cls_label = (torch.zeros(1, 1) + slot).cuda() # torch.Size([1, 1])
                dom_label = (torch.zeros(1, 1) + self.name2domId[datasetname]).cuda() # torch.Size([1, 1])
                self.feat_vecs = torch.cat((self.feat_vecs, cls_vec.unsqueeze(dim=0)), dim=0) # torch.Size([1, 7])
                self.feat_vec_labels = torch.cat((self.feat_vec_labels, cls_label), dim=0) # torch.Size([1, 1])
                self.feat_vec_domlabels = torch.cat((self.feat_vec_domlabels, dom_label), dim=0) #torch.Size([0])

    def draw_tsne(self, domains2draw, adding_name=None, plot_memory=False, clscolor=True):
        feat_vecs_temp = F.normalize(self.feat_vecs.clone(), dim=1).cpu().numpy()
        feat_vec_labels_temp = self.feat_vec_labels.clone().to(torch.int64).squeeze().cpu().numpy()
        feat_vec_domlabels_temp = self.feat_vec_domlabels.clone().to(torch.int64).squeeze().cpu().numpy()

        if self.mem_vecs is not None and plot_memory:
            mem_vecs_temp = self.mem_vecs.clone().cpu().numpy()
            mem_vec_labels_temp = self.mem_vec_labels.clone().cpu().numpy()

        if adding_name is not None:
            tsne_file_name = adding_name+'_feature_tsne_among_' + ''.join(domains2draw) + '_' + str(self.perplexity) + '_' + str(self.learning_rate)
        else:
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(self.perplexity) + '_' + str(self.learning_rate)
        tsne_file_name = os.path.join(self.tsne_path,tsne_file_name)

        if clscolor:
            sequence_of_colors = np.array([list(self.trainId2color[x]) for x in range(19)])/255.0
        else:
            sequence_of_colors = ["tab:purple", "tab:pink", "lightgray","dimgray","yellow","tab:brown","tab:orange","blue","tab:green","darkslategray","tab:cyan","tab:red","lime","tab:blue","navy","tab:olive","blueviolet", "deeppink","red"]
            sequence_of_colors[1] = "tab:olive"
            sequence_of_colors[2] = "tab:grey"
            sequence_of_colors[5] = "tab:cyan"
            sequence_of_colors[8] =  "tab:pink"
            sequence_of_colors[10] = "tab:brown"
            sequence_of_colors[13] = "tab:red"

        name2domId = {self.domId2name[x] : x for x in self.domId2name.keys()}
        domIds2draw = [name2domId[x] for x in domains2draw]
        name2trainId = {v:k for k,v in self.trainId2name.items()}
        trainIds2draw = [name2trainId[x] for x in self.selected_cls]
        domain_color = ["tab:blue", "tab:green","tab:orange","tab:purple","black"]
        assert len(feat_vec_domlabels_temp.shape) == 1
        assert len(feat_vecs_temp.shape) == 2
        assert len(feat_vec_labels_temp.shape) == 1

        # domain spliting
        dom_idx = np.array([x in domIds2draw for x in feat_vec_domlabels_temp])
        feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = feat_vecs_temp[dom_idx, :], feat_vec_labels_temp[dom_idx], \
                                                                       feat_vec_domlabels_temp[dom_idx]

        # max_pointnum random sampling.
        if feat_vecs_temp.shape[0] > self.max_pointnum:
            pointnum_predraw = feat_vec_labels_temp.shape[0]
            dom_idx = np.random.randint(0,pointnum_predraw,self.max_pointnum)
            feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = feat_vecs_temp[dom_idx, :], feat_vec_labels_temp[dom_idx], feat_vec_domlabels_temp[dom_idx]

        if self.mem_vecs is not None and plot_memory:
            mem_address = feat_vecs_temp.shape[0]
            vecs2tsne = np.concatenate((feat_vecs_temp,mem_vecs_temp))
        else:
            vecs2tsne = feat_vecs_temp

        for tries in range(self.duplication):
            X_embedded = self.TSNE.fit_transform(vecs2tsne)
            print('\ntsne done')
            X_embedded[:,0] = (X_embedded[:,0] - X_embedded[:,0].min()) / (X_embedded[:,0].max() - X_embedded[:,0].min())
            X_embedded[:,1] = (X_embedded[:,1] - X_embedded[:,1].min()) / (X_embedded[:,1].max() - X_embedded[:,1].min())

            if self.mem_vecs is not None and plot_memory:
                feat_coords = X_embedded[:mem_address,:]
                mem_coords = X_embedded[mem_address:,:]
            else:
                feat_coords = X_embedded

            ##### color means class
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            for dom_i in domIds2draw:
                for cls_i in trainIds2draw:
                    temp_coords = feat_coords[(feat_vec_labels_temp == cls_i) & (feat_vec_domlabels_temp == dom_i),:]
                    ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                               color=sequence_of_colors[cls_i], label=self.domId2name[dom_i]+'_'+self.trainId2name[cls_i], s=20, marker = 'x')

            if self.mem_vecs is not None and plot_memory:
                for cls_i in trainIds2draw:
                    ax.scatter(mem_coords[mem_vec_labels_temp == cls_i, 0], mem_coords[mem_vec_labels_temp == cls_i, 1],
                               color=sequence_of_colors[cls_i], label='mem_' + str(self.trainId2name[cls_i]), s=100, marker="^",edgecolors = 'black')

            print('scatter plot done')
            # lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            tsne_file_path = tsne_file_name+'_'+str(tries)+'_colorclass'+self.extention
            fig.savefig(tsne_file_path)

            # fig.savefig(tsne_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
            # plt.show()
            fig.clf()

            ##### color means domains
            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(111)
            #
            # for dom_i in domIds2draw:
            #     for cls_i in trainIds2draw:
            #         temp_coords = feat_coords[(feat_vec_labels_temp == cls_i) & (feat_vec_domlabels_temp == dom_i),:]
            #         ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
            #                    color= domain_color[dom_i], label=self.domId2name[dom_i]+'_'+self.trainId2name[cls_i], s=20, marker = 'x')
            #
            # if self.mem_vecs is not None and plot_memory:
            #     for cls_i in trainIds2draw:
            #         ax.scatter(mem_coords[mem_vec_labels_temp == cls_i, 0], mem_coords[mem_vec_labels_temp == cls_i, 1],
            #                    color=sequence_of_colors[cls_i], label='mem_' + str(self.trainId2name[cls_i]), s=100, marker="^",edgecolors = 'black')
            #
            # print('scatter plot done')
            # lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
            # ax.set_xlim(-0.05, 1.05)
            # ax.set_ylim(-0.05, 1.05)
            # tsne_file_path = tsne_file_name+'_'+str(tries)+'_colordomain'+self.extention
            # fig.savefig(tsne_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
            # # plt.show()
            # fig.clf()
            #
            # # print memory coordinate
            # if self.mem_vecs is not None and plot_memory:
            #     print("memory coordinates")
            #     for i,x in enumerate(mem_vec_labels_temp):
            #         print(mem_coords[i,:],self.trainId2name[x])
        return tsne_file_path

if __name__ == '__main__':
    # all_class = False   # t-SNE展示全部类别，还是部分类别
    selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light']
    # if all_class:
    #     selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
    #                     'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    # else:
    #     selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light']
        # 自己指定要进行t-SNE的类别（可以根据t-SNE的效果选择最好的几个类别即可）

    domId2name = {
        0:'gtav',
        1:'synthia',
        2:'cityscapes',
        3:'bdd100k',
        4:'mapillary',
        5:'idd',
        6: 'LoveDA'
        }
    # 为每个数据集指定一个ID

    # 默认使用cityscapes里面的标签类别
    # import cityscapes_labels
    # trainId2name = cityscapes_labels.trainId2name
    trainId2name = {255: 'trailer',
                    0: 'road',
                    1: 'sidewalk',
                    2: 'building',
                    3: 'wall',
                    4: 'fence',
                    5: 'pole',
                    6: 'traffic light',
                    7: 'traffic sign',
                    8: 'vegetation',
                    9: 'terrain',
                    10: 'sky',
                    11: 'person',
                    12: 'rider',
                    13: 'car',
                    14: 'truck',
                    15: 'bus',
                    16: 'train',
                    17: 'motorcycle',
                    18: 'bicycle',
                    -1: 'license plate'}
    # trainId2color = cityscapes_labels.trainId2color
    trainId2color = {255: (0, 0, 110),
                    0: (0,0,0),
                    1: (254, 0, 0),
                    2: (255, 255, 0),
                    3: (0, 0, 254),
                    4: (160, 130, 184),
                    5: (1, 255, 1),
                    6: (255, 196, 130),
                    7: (250, 170, 30),
                    8: (220, 220, 0),
                    9: (107, 142, 35),
                    10: (70, 130, 180),
                    11: (220, 20, 60),
                    12: (255, 0, 0),
                    13: (0, 0, 142),
                    14: (0, 0, 70),
                    15: (0, 60, 100),
                    16: (0, 80, 100),
                    17: (0, 0, 230),
                    18: (119, 11, 32),
                    -1: (0, 0, 143)}

    output_dir = './'
    tsnecuda = True
    extention = '.png'
    duplication = 10
    plot_memory = False
    clscolor = True
    domains2draw = ['LoveDA']
    # 指定需要进行t-SNE的域，即数据集

    tsne_runner = RunTsne(selected_cls=selected_cls,
                          domId2name=domId2name,
                          trainId2name=trainId2name,
                          trainId2color=trainId2color,
                          output_dir=output_dir,
                          tsnecuda=tsnecuda,
                          extention=extention,
                          duplication=duplication)

    ################ inference过程 ################
    # 注意这里是伪代码，根据自己的情况进行修改
    from module.Encoder import Deeplabv2Test
    cfg = import_config('ast.2rural')
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
    model.eval()
    ckpt_path = '/home/zjb/PycharmProjects/uda/log/ast/2rural/RURAL8000.pth'
    model_state_dict = torch.load(ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)
    eval_dataloader = LoveDALoader(cfg.EVAL_DATA_CONFIG)
    with torch.no_grad():
        for ret, ret_gt in tqdm(eval_dataloader):
            ret = ret.to(torch.device('cuda'))
            B, C, H, W = ret.shape
            gt_image= ret_gt['cls'].view(-1,H,W)
            gt_cuda = gt_image.cuda() # torch.Size([1, 1024, 1024])
            features = model(ret)[1]# torch.Size([1, 2048, 1024, 1024])

            dataset = 'LoveDA'
            tsne_runner.input2basket(features, gt_cuda, dataset)

    tsne_runner.draw_tsne(domains2draw, plot_memory=plot_memory, clscolor=clscolor)

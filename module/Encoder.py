import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class PPMBilinear(nn.Module):
    def __init__(self, num_classes=7, fc_dim=2048,
                 use_aux=False, pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d
                 ):
        super(PPMBilinear, self).__init__()
        self.use_aux = use_aux
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        if self.use_aux:
            self.cbr_deepsup = nn.Sequential(
                nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(fc_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        # conv5 = conv_out[-1]
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))

        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_aux and self.training:
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            return x
        else:
            return x


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class SEBlock(nn.Module):
    def __init__(self, inplanes, r=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            nn.Linear(inplanes, inplanes // r),
            nn.ReLU(inplace=True),
            nn.Linear(inplanes // r, inplanes),
            nn.Sigmoid()
        )

    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)


class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate=0.1, use_se=True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
            nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                nn.ReLU(inplace=True)]))

        for dilation, padding in zip(dilation_series, padding_series):
            # self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                    # nn.ReflectionPad2d(padding),
                    nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                    nn.GroupNorm(num_groups=32, num_channels=256, affine=True),
                    nn.ReLU(inplace=True)]))

        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                                              nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1,
                                                        padding=1, dilation=1, bias=True),
                                              nn.GroupNorm(num_groups=32, num_channels=256, affine=True)])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1,
                          bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine=True)])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
                                    nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False)])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m,
                                                                                                 nn.GroupNorm) or isinstance(
                    m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=True):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat((out, self.conv2d_list[i + 1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            # out = self.head(out)
            out = out
            return out


from module.resnet import ResNetEncoder
import ever as er


class Deeplabv2(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2, self).__init__(config)
        self.objective_vectors = torch.zeros([self.config.num_classes, 256])
        self.objective_vectors_num = torch.zeros([self.config.num_classes])
        self.class_distribution = torch.zeros([self.config.num_classes])
        self.prototypes = Variable(torch.zeros([self.config.num_classes, 256]), requires_grad=True).cuda()
        self.encoder = ResNetEncoder(self.config.backbone)

        if self.config.multi_layer:
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module2, self.config.inchannels // 2,
                                                        [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                      [6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.config.multi_layer:
            if self.config.cascade:
                c3, c4 = self.encoder(x)[-2:]  # torch.Size([8, 1024, 32, 32]); torch.Size([8, 2048, 32, 32])
                x1 = self.layer5(c3)  # torch.Size([8, 7, 32, 32])
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)  # torch.Size([8, 7, 512, 512])
                x2 = self.layer6(c4)  # torch.Size([8, 7, 32, 32])

                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x2).softmax(dim=1)
            else:
                x = self.encoder(x)[-1]
                x1 = self.layer5(x)
                x2 = self.layer6(x)
                if self.training:
                    return x1, x2, x
                else:
                    x1_out = x1['out']
                    x1_feat = x1['feat']
                    x2_out = x2['out']
                    x2_feat = x2['feat']
                    x1 = F.interpolate(x1_out, (H, W), mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2_out, (H, W), mode='bilinear', align_corners=True)
                    return (x1 + x2).softmax(dim=1)

        else:
            feat, x = self.encoder(x)[-2:]
            # x = self.layer5(x)

            x = self.cls_pred(x)
            # x = self.cls_pred(x)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
            # feat = F.interpolate(feat, (H, W), mode='bilinear', align_corners=True)
            if self.training:
                return x, feat
            else:
                return x.softmax(dim=1), feat

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm=False,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,

            ),
            inchannels=2048,
            num_classes=7
        ))

    @torch.no_grad()
    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        #     dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q *= self.class_distribution.unsqueeze(1)

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        # Q = torch.argmax(Q, 0)
        return Q.t()

    def get_prototype_weight(self, feat):
        bs, _, h, w = feat.shape
        feat = feat.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 256)
        feat = torch.nn.functional.normalize(feat, dim=1, p=2)
        proto = torch.nn.functional.normalize(self.prototypes, dim=1, p=2)  # [7,256]
        target = torch.mm(feat, proto.t()).view(bs, h, w, self.config.num_classes).transpose(3, 2).transpose(2, 1)
        weight = F.softmax(target / 0.8, dim=1)  # torch.Size([4, 7, 32, 256])
        return weight

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - 0.0001) + 0.0001 * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))


class Deeplabv2Test(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2Test, self).__init__(config)
        self.objective_vectors = torch.zeros([self.config.num_classes, 256])
        self.objective_vectors_num = torch.zeros([self.config.num_classes])
        self.class_distribution = torch.zeros([self.config.num_classes])
        self.prototypes = Variable(torch.zeros([self.config.num_classes, 256]), requires_grad=True).cuda()
        self.encoder = ResNetEncoder(self.config.backbone)

        if self.config.multi_layer:
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module2, self.config.inchannels // 2,
                                                        [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module2, self.config.inchannels, [6, 12, 18, 24],
                                                      [6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.config.multi_layer:
            if self.config.cascade:
                c3, c4 = self.encoder(x)[-2:]  # torch.Size([8, 1024, 32, 32]); torch.Size([8, 2048, 32, 32])
                x1 = self.layer5(c3)  # torch.Size([8, 7, 32, 32])
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)  # torch.Size([8, 7, 512, 512])
                x2 = self.layer6(c4)  # torch.Size([8, 7, 32, 32])

                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x2).softmax(dim=1)
            else:
                x = self.encoder(x)[-1]
                x1 = self.layer5(x)
                x2 = self.layer6(x)
                if self.training:
                    return x1, x2, x
                else:
                    x1_out = x1['out']
                    x1_feat = x1['feat']
                    x2_out = x2['out']
                    x2_feat = x2['feat']
                    # x1 = F.interpolate(x1_out, (H, W), mode='bilinear', align_corners=True)
                    # x2 = F.interpolate(x2_out, (H, W), mode='bilinear', align_corners=True)
                    return (x1_out+x2_out)/2, (x1_feat+x2_feat)/2

        else:
            feat, x = self.encoder(x)[-2:]
            # x = self.layer5(x)

            x = self.cls_pred(x)
            # x = self.cls_pred(x)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
            # feat = F.interpolate(feat, (H, W), mode='bilinear', align_corners=True)
            if self.training:
                return x, feat
            else:
                return x.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm=False,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,

            ),
            inchannels=2048,
            num_classes=7
        ))

    @torch.no_grad()
    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        #     dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(3):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q *= self.class_distribution.unsqueeze(1)

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        # Q = torch.argmax(Q, 0)
        return Q.t()

    def get_prototype_weight(self, feat):
        bs, _, h, w = feat.shape
        feat = feat.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 256)
        feat = torch.nn.functional.normalize(feat, dim=1, p=2)
        proto = torch.nn.functional.normalize(self.prototypes, dim=1, p=2)  # [7,256]
        target = torch.mm(feat, proto.t()).view(bs, h, w, self.config.num_classes).transpose(3, 2).transpose(2, 1)
        weight = F.softmax(target / 0.8, dim=1)  # torch.Size([4, 7, 32, 256])
        return weight

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - 0.0001) + 0.0001 * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))
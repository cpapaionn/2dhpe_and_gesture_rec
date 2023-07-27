# encoding: utf-8
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from HumanPose2D.config import config as cfg
from base_model import resnet18, resnet50
from seg_opr.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion, AttentionSynapses
from HumanPose2D.pix2pix_networks import define_G, define_D

class CNN_GAN_AS(nn.Module):
    def __init__(self, out_planes, is_training, pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(CNN_GAN_AS, self).__init__()

        self.output_scale = cfg.MODEL_OUTPUT_SCALE
        self.is_training = is_training
        self.is_deep = cfg.MODEL_DEEP

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        i2i_arms = [AttentionRefinement(256, 128, norm_layer),
                    AttentionRefinement(128, 128, norm_layer)]

        i2i_atts = [AttentionSynapses(256, 128, norm_layer),
                    AttentionSynapses(256, 128, norm_layer)]

        conv_channel = 128

        if self.is_deep:
            ch_depth = 2048
        else: 
            ch_depth = 512
        self.global_context = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                            ConvBnRelu(ch_depth, conv_channel, 1, 1, 0,
                                                       has_bn=True,
                                                       has_relu=True, 
                                                       has_bias=False, 
                                                       norm_layer=norm_layer)
                                           )

        arms = [AttentionRefinement(ch_depth, conv_channel, norm_layer),
                AttentionRefinement(ch_depth // 2, conv_channel, norm_layer)]
     
        refines = [ConvBnRelu(conv_channel*2, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel*2, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]

        if self.is_training:
            heads = [SegHead(conv_channel, out_planes, self.output_scale * 2, True, norm_layer),
                     SegHead(conv_channel, out_planes, self.output_scale, True, norm_layer),
                     SegHead(conv_channel * 2, out_planes, self.output_scale, False, norm_layer)]
            self.i2i_seg_head = I2ISegHead(64, cfg.num_classes, self.output_scale)

        else:
            heads = [None, 
                     None, 
                     SegHead(conv_channel * 2, out_planes, self.output_scale, False, norm_layer)]
            self.i2i_seg_head = None

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2, 1, norm_layer)
        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)
        self.i2i_arms = nn.ModuleList(i2i_arms)
        self.i2i_atts = nn.ModuleList(i2i_atts)

        if self.is_training:
            self.init_weights()

        self.i2i_path = define_G(input_nc=ch_depth, output_nc=3, ngf=64, netG='resnet_4blocks', norm='batch',
                                 use_dropout=True, init_type='normal', init_gain=0.02, is_training=self.is_training, 
                                 gpu_ids=cfg.GPUS)

        if self.is_deep:
            self.context_path = resnet50(pretrained_model, norm_layer=norm_layer,
                                         bn_eps=cfg.bn_eps,
                                         bn_momentum=cfg.bn_momentum,
                                         deep_stem=True, stem_width=64)
        else:
            self.context_path = resnet18(pretrained_model, norm_layer=norm_layer,
                                         bn_eps=cfg.bn_eps,
                                         bn_momentum=cfg.bn_momentum,
                                         deep_stem=False, stem_width=64)

        if self.is_training:
            self.netD = define_D(input_nc=6, ndf=32, netD='basic', n_layers_D=2, norm='batch', init_type='normal',
                                 init_gain=0.02, gpu_ids=cfg.GPUS)

    def forward(self, data):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        i2i_blocks = self.i2i_path(context_blocks[0])

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, i2i_arm, i2i_att, refine) in enumerate(zip(context_blocks[:2], self.arms, self.i2i_arms,
                                                                    self.i2i_atts, self.refines)):
            fm = arm(fm)
            i2i = i2i_arm(i2i_blocks[i])
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]), mode='bilinear', align_corners=True)
            last_fm_att = i2i_att(last_fm, i2i)
            last_fm = torch.cat((last_fm, last_fm_att), dim=1)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)

        if not self.is_training:
            return self.heads[-1](concate_fm)

        pred_out.append(concate_fm)
        pred_out.append(i2i_blocks[-2])
        pred_out.append(i2i_blocks[-1])

        # segmentation branch
        aux0_output = self.heads[0](pred_out[0])
        aux1_output = self.heads[1](pred_out[1])
        main_output = self.heads[-1](pred_out[2])

        # i2i branch
        aux2_output = self.i2i_seg_head(pred_out[-2])
        if self.output_scale > 1:
            i2i_output = F.interpolate(pred_out[-1], scale_factor=self.output_scale, mode='bilinear', align_corners=True)
        else:
            i2i_output = pred_out[-1]

        return aux0_output, aux1_output, main_output, aux2_output, i2i_output

    def forward_D(self, data_resized, i2i_label, i2i_output):
        fake_pair = torch.cat((data_resized, i2i_output), 1)
        pred_fake, classes_fake = self.netD(fake_pair.detach())

        real_pair = torch.cat((data_resized, i2i_label), 1)
        pred_real, classes_real = self.netD(real_pair)

        return pred_fake, classes_fake, pred_real, classes_real

    def forward_i2i_net(self, data_resized, i2i_output):
        fake_pair = torch.cat((data_resized, i2i_output), 1)
        pred_fake, classes_fake = self.netD(fake_pair)

        return pred_fake, classes_fake

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class SegHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(SegHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        return output


class I2ISegHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(I2ISegHead, self).__init__()

        self.scale = scale

        self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                  stride=1, padding=0)

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)

        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        return output


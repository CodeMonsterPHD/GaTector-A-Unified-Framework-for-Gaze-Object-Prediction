from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn

from lib.nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class Resnet(nn.Module):
    def __init__(self, phi, pretrained=False):
        super(Resnet, self).__init__()
        self.edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = resnet50(pretrained)
        del model.avgpool
        del model.fc
        self.model = model
        self.ps = nn.PixelShuffle(2)

    def forward(self, image, face, train_mode):
        image = self.model.conv1(image)
        image = self.model.bn1(image)
        image = self.model.relu(image)
        image = self.model.maxpool(image)

        image = self.model.layer1(image)
        feat1 = self.model.layer2(image)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)

        pix1 = self.ps(feat1)
        pix2 = self.ps(feat2)
        pix3 = self.ps(feat3)

        face = self.model.conv2(face)
        face = self.model.bn2(face)
        face = self.model.relu(face)
        face = self.model.maxpool(face)

        face = self.model.layer1(face)
        face = self.model.layer2(face)
        face = self.model.layer3(face)
        face = self.model.layer4(face)

        image = self.model.layer5_scene(feat3)
        face = self.model.layer5_face(face)
        if train_mode == 0:
            return image, face, pix1, pix2, pix3
        else:
            return feat1, feat2, feat3


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class Pixel_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pixel_shuffle, self).__init__()

        self.pixel_shuffle = nn.Sequential(
            nn.PixelShuffle(2),
            conv2d(in_channels, out_channels, 3)
        )

    def forward(self, x, ):
        x = self.pixel_shuffle(x)
        return x


def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m


class GaTectorBody(nn.Module):
    def __init__(self, anchors_mask, num_classes,train_mode):
        super(GaTectorBody, self).__init__()

        self.backbone = Resnet(phi='resnet50', pretrained=True)
        # self.backbone = darknet53(None)
        if train_mode==0:
            self.conv1 = make_three_conv([512, 1024], 512)
            self.SPP = SpatialPyramidPooling()
            self.conv2 = make_three_conv([512, 1024], 2048)

            self.upsample1 = Upsample(512, 256)
            self.conv_for_P4 = conv2d(256, 256, 1)
            self.make_five_conv1 = make_five_conv([256, 512], 512)

            self.upsample2 = Upsample(256, 128)
            self.conv_for_P3 = conv2d(128, 128, 1)
            self.make_five_conv2 = make_five_conv([128, 256], 256)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

            self.down_sample1 = conv2d(128, 256, 3, stride=2)
            self.make_five_conv3 = make_five_conv([256, 512], 512)

            self.down_sample2 = conv2d(256, 512, 3, stride=2)
            self.make_five_conv4 = make_five_conv([512, 1024], 1024)

            # Pixle_shuffle
            self.ps1 = Pixel_shuffle(128, 256)
            self.ps2 = Pixel_shuffle(64, 128)
        else:
            self.conv1 = make_three_conv([512, 1024], 2048)
            self.SPP = SpatialPyramidPooling()
            self.conv2 = make_three_conv([512, 1024], 2048)

            self.upsample1 = Upsample(512, 256)
            self.conv_for_P4 = conv2d(1024, 256, 1)
            self.make_five_conv1 = make_five_conv([256, 512], 512)

            self.upsample2 = Upsample(256, 128)
            self.conv_for_P3 = conv2d(512, 128, 1)
            self.make_five_conv2 = make_five_conv([128, 256], 256)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head3 = yolo_head([256, len(anchors_mask[0]) * (5 + num_classes)], 128)

            self.down_sample1 = conv2d(128, 256, 3, stride=2)
            self.make_five_conv3 = make_five_conv([256, 512], 512)

            # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
            self.yolo_head2 = yolo_head([512, len(anchors_mask[1]) * (5 + num_classes)], 256)

            self.down_sample2 = conv2d(256, 512, 3, stride=2)
            self.make_five_conv4 = make_five_conv([512, 1024], 1024)

            # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
            self.yolo_head1 = yolo_head([1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

        # GOO network
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # head pathway
        self.conv_face_scene = nn.Conv2d(2560, 2048, kernel_size=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU()
        )
        # attention
        self.attn = nn.Linear(1808, 1 * 7 * 7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, images, head, face,train_mode):

        if train_mode==0:
            #  backbone
            scene_feat, face_feat, x2, x1, x0 = self.backbone(images, face, train_mode)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
            P5 = self.conv1(x0)
            testP5=nn.MaxPool2d(kernel_size=5,stride=1,padding=5//2)
            P5 = self.SPP(P5)
            # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.conv2(P5)

            # 13,13,512 -> 13,13,256 -> 26,26,256
            P5_upsample = self.ps1(P5)  # self.upsample1(P5)
            # 26,26,512 -> 26,26,256
            P4 = self.conv_for_P4(x1)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P4, P5_upsample], axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv1(P4)

            # 26,26,256 -> 26,26,128 -> 52,52,128
            P4_upsample = self.ps2(P4)  # self.upsample2(P4)
            # 52,52,256 -> 52,52,128
            P3 = self.conv_for_P3(x2)
            # 52,52,128 + 52,52,128 -> 52,52,256
            P3 = torch.cat([P3, P4_upsample], axis=1)
            # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
            P3 = self.make_five_conv2(P3)

            out2 = self.yolo_head3(P3)

            #GOO
            #reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
            head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
            # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
            # get and reshape attention weights such that it can be multiplied with scene feature map
            # xz=torch.cat(head_reduced, face_feat_reduced)
            # x0=self.attn(x,1)
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            attn_weights = attn_weights.view(-1, 1, 49)
            attn_weights = F.softmax(attn_weights, dim=2)  # soft attention weights single-channel
            attn_weights = attn_weights.view(-1, 1, 7, 7)

            head_conv = self.conv_block(head)

            # attn_weights = torch.ones(attn_weights.shape)/49.0
            scene_feat = torch.cat((scene_feat, head_conv), 1)
            attn_applied_scene_feat = torch.mul(attn_weights,
                                                scene_feat)  # (N, 1, 7, 7) # applying attention weights on scene feat

            scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)
            scene_face_feat = self.conv_face_scene(scene_face_feat)

            # scene + face feat -> encoding -> decoding
            encoding = self.compress_conv1(scene_face_feat)
            encoding = self.compress_bn1(encoding)
            encoding = self.relu(encoding)
            encoding = self.compress_conv2(encoding)
            encoding = self.compress_bn2(encoding)
            encoding = self.relu(encoding)
            x = self.deconv1(encoding)
            x = self.deconv_bn1(x)
            x = self.relu(x)
            x = self.deconv2(x)
            x = self.deconv_bn2(x)
            x = self.relu(x)
            x = self.deconv3(x)
            x = self.deconv_bn3(x)
            x = self.relu(x)
            x = self.conv4(x)

            return x, out2
        else:
            #  backbone
            x2, x1, x0 = self.backbone(images, face, train_mode)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048
            P5 = self.conv1(x0)
            P5 = self.SPP(P5)
            # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.conv2(P5)

            # 13,13,512 -> 13,13,256 -> 26,26,256
            P5_upsample = self.upsample1(P5)
            # 26,26,512 -> 26,26,256
            P4 = self.conv_for_P4(x1)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P4, P5_upsample], axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv1(P4)

            # 26,26,256 -> 26,26,128 -> 52,52,128
            P4_upsample = self.upsample2(P4)
            # 52,52,256 -> 52,52,128
            P3 = self.conv_for_P3(x2)
            # 52,52,128 + 52,52,128 -> 52,52,256
            P3 = torch.cat([P3, P4_upsample], axis=1)
            # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
            P3 = self.make_five_conv2(P3)

            # 52,52,128 -> 26,26,256
            P3_downsample = self.down_sample1(P3)
            # 26,26,256 + 26,26,256 -> 26,26,512
            P4 = torch.cat([P3_downsample, P4], axis=1)
            # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
            P4 = self.make_five_conv3(P4)

            # 26,26,256 -> 13,13,512
            P4_downsample = self.down_sample2(P4)
            # 13,13,512 + 13,13,512 -> 13,13,1024
            P5 = torch.cat([P4_downsample, P5], axis=1)
            # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
            P5 = self.make_five_conv4(P5)

            # ---------------------------------------------------#
            #   第三个特征层
            #   y3=(batch_size,75,52,52)
            # ---------------------------------------------------#
            out2 = self.yolo_head3(P3)
            # ---------------------------------------------------#
            #   第二个特征层
            #   y2=(batch_size,75,26,26)
            # ---------------------------------------------------#
            out1 = self.yolo_head2(P4)
            # ---------------------------------------------------#
            #   第一个特征层
            #   y1=(batch_size,75,13,13)
            # ---------------------------------------------------#
            out0 = self.yolo_head1(P5)

            x=0

            return x, out0, out1, out2



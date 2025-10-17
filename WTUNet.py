#!/usr/bin/env python
# -*- coding:utf-8 -*-


# import module your need
import torch
import torch.nn as nn
from models.WTUNet.blocks import DoubleConv, WTDown, WTUp, OutConv


class WTUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(WTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # input: B, 1, 128, 128
        self.inc = (DoubleConv(n_channels, 64)) # B, 64, 128, 128
        self.down1 = (WTDown(64, 128)) # B, 128, 64, 64
        self.down2 = (WTDown(128, 256)) # B, 256, 32, 32
        self.down3 = (WTDown(256, 512)) # B, 512, 16, 16
        factor = 2 if bilinear else 1
        self.down4 = (WTDown(512, 1024 // factor)) # B, 512, 8, 8
        self.up1 = (WTUp(1024, 512 // factor, bilinear)) # B, 512, 16, 16
        self.up2 = (WTUp(512, 256 // factor, bilinear)) # B, 256, 32, 32
        self.up3 = (WTUp(256, 128 // factor, bilinear)) # B, 128, 64, 64
        self.up4 = (WTUp(128, 64, bilinear)) # B, 64, 128, 128
        self.outc = (OutConv(64, n_classes)) # B, 1, 128, 128

        self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x) # B, 64, 128, 128
        x2, h2 = self.down1(x1)
        x3, h3 = self.down2(x2)
        x4, h4 = self.down3(x3)
        x5, h5 = self.down4(x4) # intput: B, 256, 16, 16, output: B, 512, 8, 8
        x = self.up1(x5, h5, x4) # input: B, 512, 8, 8, ouput: B, 256, 16, 16
        x = self.up2(x, h4, x3)
        x = self.up3(x, h3, x2)
        x = self.up4(x, h2, x1)
        logits = self.outc(x)
        return_dict = {'embeddings': x5, 'logits': logits}
        return return_dict

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = WTUNet(1, 1)
    x = torch.randn(1, 1, 128, 128)
    print(model(x).shape)
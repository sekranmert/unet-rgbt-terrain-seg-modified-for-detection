import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
            

    def forward(self, x1, x2):

        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # x1 has lower spatial resolution, we need to pad it
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        return self.conv(x)


class BottleneckFuseCat(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):

        # Concatenate along channel dimension
        x = torch.cat([x1, x2], dim=1)  

        return self.conv_block(x)

class BottleneckFuseAdd(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):

        x = x1 + x2

        return self.conv_block(x)

class FuseConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(FuseConv, self).__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):

        x = torch.cat([x1, x2], dim=1)

        return self.fuse_conv(x)    

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.first = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1   = Up(1024, 512)
        self.up2   = Up(512, 256)
        self.up3   = Up(256, 128)
        self.up4   = Up(128, 64)

        self.out  = OutConv(64, n_classes)

    def forward(self, x):
        
        x1 = self.first(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.out(x)

        return logits    

class PairUNet(nn.Module):

    def __init__(self, rgb_channels, ir_channels, n_classes):

        super(PairUNet, self).__init__()

        self.rgb_channels = rgb_channels
        self.ir_channels = ir_channels
        self.n_classes = n_classes
        

        self.first_rgb = DoubleConv(rgb_channels, 64)
        self.first_ir  = DoubleConv(ir_channels, 64)

        self.down_rgb_1 = Down(64, 128)
        self.down_rgb_2 = Down(128, 256)
        self.down_rgb_3 = Down(256, 512)
        self.down_rgb_4 = Down(512, 1024)
        
        self.down_ir_1 = Down(64, 128)
        self.down_ir_2 = Down(128, 256)
        self.down_ir_3 = Down(256, 512)
        self.down_ir_4 = Down(512, 1024)

        self.bottleneck_conv = BottleneckFuseCat(2048, 1024)

        self.fuse_conv_1 = FuseConv(128, 64)
        self.fuse_conv_2 = FuseConv(256, 128)
        self.fuse_conv_3 = FuseConv(512, 256)
        self.fuse_conv_4 = FuseConv(1024, 512)
        
        self.up1  = Up(1024, 512)
        self.up2  = Up(512, 256)
        self.up3  = Up(256, 128)
        self.up4  = Up(128, 64)

        self.out = OutConv(64, n_classes)

    def forward(self, x_rgb, x_ir):

        x_rgb_1 = self.first_rgb(x_rgb)     # 64 x 600 x 900
        x_rgb_2 = self.down_rgb_1(x_rgb_1) # 128 x 300 x 480
        x_rgb_3 = self.down_rgb_2(x_rgb_2) # 256 x 150 x 240
        x_rgb_4 = self.down_rgb_3(x_rgb_3) # 512 x 75 x 120
        x_rgb_5 = self.down_rgb_4(x_rgb_4) # 1024 x 37 x 60

        x_ir_1 = self.first_ir(x_ir)       # 64 x 600 x 900
        x_ir_2 = self.down_ir_1(x_ir_1)    # 128 x 300 x 480
        x_ir_3 = self.down_ir_2(x_ir_2)    # 256 x 150 x 240
        x_ir_4 = self.down_ir_3(x_ir_3)    # 512 x 75 x 120
        x_ir_5 = self.down_ir_4(x_ir_4)    # 1024 x 37 x 60

        fused_5 = self.bottleneck_conv(x_rgb_5, x_ir_5)

        fused_4 = self.fuse_conv_4(x_rgb_4, x_ir_4)
        fused_3 = self.fuse_conv_3(x_rgb_3, x_ir_3)
        fused_2 = self.fuse_conv_2(x_rgb_2, x_ir_2)
        fused_1 = self.fuse_conv_1(x_rgb_1, x_ir_1)

        x = self.up1(fused_5, fused_4)
        x = self.up2(x, fused_3)
        x = self.up3(x, fused_2)
        x = self.up4(x, fused_1)

        logits = self.out(x)
        
        return logits

if __name__ == "__main__":

    import cv2
    import numpy as np
    from utils import *

    x1 = torch.randn(1, 3, 600, 960)
    x2 = torch.randn(1, 3, 600, 960)

    model = PairUNet(rgb_channels=3, ir_channels=3, n_classes=12)

    y_logits    = model(x1, x2)
    y_pred_one_hot = logit_to_one_hot(y_logits, model.n_classes)

    print(x1.shape)
    print(x2.shape)
    print(y_pred_one_hot.shape)

    # img = one_hot_to_mask(y_pred_one_hot).numpy().astype(np.float32)

    # img = cv2.normalize(
    # img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # cv2.imshow("Image", img)
    # cv2.waitKey(0)

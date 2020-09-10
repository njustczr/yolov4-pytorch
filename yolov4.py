import torch.nn as nn
from cslayers import *
import torch.backends.cudnn as cudnn
from torchsummary import summary

__all__ = ['YoloV4']

class CsDarknet53(nn.Module):
    def __init__(self, num_classes):
        super(CsDarknet53, self).__init__()

        input_channels = 32

        # Network
        self.stage1 = Conv2dBatchLeaky(3, input_channels, 3, 1)
        self.stage2 = Stage2(input_channels)
        self.stage3 = Stage3(4*input_channels)
        self.stage4 = Stage(4*input_channels, 8)
        self.stage5 = Stage(8*input_channels, 8)
        self.stage6 = Stage(16*input_channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)

        return stage6, stage5, stage4

class YoloV4(nn.Module):
    def __init__(self):
        super(YoloV4, self).__init__()

        input_channel = 1024

        self.backbone = CsDarknet53(None)
        self.conv1 = Conv2dBatchLeaky(input_channel, input_channel//2, 1, 1)
        self.conv2 = Conv2dBatchLeaky(input_channel//2, input_channel, 3, 1)
        self.conv3 = Conv2dBatchLeaky(input_channel, input_channel//2, 1, 1)
        self.spp = SpatialPyramidPooling()
        self.conv4 = Conv2dBatchLeaky(2*input_channel, input_channel//2, 1, 1)
        self.conv5 = Conv2dBatchLeaky(input_channel//2, input_channel, 3, 1)
        self.conv6 = Conv2dBatchLeaky(input_channel, input_channel//2, 1, 1)

        self.upsample1 = UpSample(input_channel//2, input_channel//4)
        self.conv7 = Conv2dBatchLeaky(input_channel//2, input_channel//4, 1, 1)
        # concat1

        # 512->256->512
        self.five_conv1 = five_ConvBL([input_channel//4, input_channel//2], input_channel//2)

        self.upsample2 = UpSample(input_channel//4, input_channel//8)
        self.conv8 = Conv2dBatchLeaky(input_channel//4, input_channel//8, 1, 1)
        # concat2

        # 256->128->256
        self.five_conv2 = five_ConvBL([input_channel//8, input_channel//4],input_channel//4)

        self.conv9 = Conv2dBatchLeaky(input_channel//8, input_channel//4, 3, 2)
        # concat3
        self.five_conv3 = five_ConvBL([input_channel//4, input_channel//2],input_channel//2)

        self.conv10 = Conv2dBatchLeaky(input_channel//4, input_channel//2, 3, 2)
        # concat4
        self.five_conv4 = five_ConvBL([input_channel//2, input_channel],input_channel)

        self.head1 = yolo_head(input_channel//8)
        self.head2 = yolo_head(input_channel//4)
        self.head3 = yolo_head(input_channel//2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0, x1, x2 = self.backbone(x)

        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        x0 = self.conv3(x0)
        x0 = self.spp(x0)
        x0 = self.conv4(x0)
        x0 = self.conv5(x0)
        p5 = self.conv6(x0)

        p5_upsample1 = self.upsample1(p5)
        x1 = self.conv7(x1)
        concat1 = torch.cat([p5_upsample1, x1], dim=1)
        concat1 = self.five_conv1(concat1)
        p5_upsample2 = self.upsample2(concat1)

        x2 = self.conv8(x2)
        concat2 = torch.cat([p5_upsample2, x2], dim=1)
        head_input1 = self.five_conv2(concat2)

        conv1 = self.conv9(head_input1)
        concat3 = torch.cat([concat1, conv1], dim=1)
        head_input2 = self.five_conv3(concat3)

        conv2 = self.conv10(head_input2)
        concat4 = torch.cat([p5,conv2],dim=1)
        head_input3 = self.five_conv4(concat4)

        return self.head1(head_input1), self.head2(head_input2), self.head3(head_input3)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    yolov4 = YoloV4()
    with torch.no_grad():
        yolov4.eval()
        data = torch.rand(1, 3, 416, 416)
        try:
            #print(darknet)
            summary(yolov4,(3,416,416))
            r1,r2,r3 = yolov4(data)
            print(r1.shape, r2.shape, r3.shape)
        except Exception as e:
            print(e)

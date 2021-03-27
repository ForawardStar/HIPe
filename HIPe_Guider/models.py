import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           RESNET
##############################

class BasicBlock(nn.Module):
    def __init__(self, type, inplane, outplane, stride):
        super(BasicBlock, self).__init__()
        conv_block = []
        if type == "Conv":
            conv_block += [nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)]
        elif type == "Deconv":
            conv_block += [nn.ConvTranspose2d(inplane, outplane, kernel_size=4, stride=stride, padding=1)]

        conv_block += [nn.BatchNorm2d(outplane),
                       nn.ReLU()]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, dilation, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dilation, norm_layer, use_dropout, use_bias):
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dilation):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3, padding = dilation-1, dilation=dilation),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        #self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('norm', nn.InstanceNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorResNet, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.nfc = 32
        self.min_nfc = 32
        self.num_layer = 6
        N = self.nfc

        self.head = ConvBlock(in_channels, N, 3, 1, 1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        mark = 0
        for i in range(self.num_layer - 2):
            block = ResidualBlock(32,1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail_state1 = nn.Sequential(
            ResidualBlock(32,1),
            ResidualBlock(32,1),
            ResidualBlock(32,2),
            ResidualBlock(32,2),
            ResidualBlock(32,4),
            ResidualBlock(32,4),
            ResidualBlock(32,8),
            ResidualBlock(32,8),
            ResidualBlock(32,1),
            #ResidualBlock(32,1),
        )

        self.tail_mask1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.tail_mask = nn.Sequential(
        #     nn.Conv2d(out_channels * 2, 64, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.InstanceNorm2d(64),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

        #for p in self.parameters():
        #    p.requires_grad = False

        #self.pretrained_dict = torch.load("saved_models/thirdstage/G_AB_250.pth")
        # self.pretrained_dict = torch.load("saved_models/secondstage/G_AB_50.pth")
        #
        # self.model_dict = self.state_dict()
        # self.model_dict.update(self.pretrained_dict)
        # self.load_state_dict(self.model_dict)

        for k,v in self.named_parameters():
            print("k: {},  requires_grad: {}".format(k,v.requires_grad))

    def forward(self, x):
        input_curr = x
        mask_input_curr = torch.ones(input_curr.shape).cuda()

        outputs = []
        masks = []
        #for step in range(4):
        x1 = self.head(input_curr)
        x2 = self.body(x1)
        del x1, mask_input_curr
        state_curr1 = self.tail_state1(x2.detach())
        mask_output_curr_1c_1 = self.tail_mask1(state_curr1)
        mask_output_curr1 = torch.cat([mask_output_curr_1c_1, mask_output_curr_1c_1, mask_output_curr_1c_1], dim=1)
        masks.append(mask_output_curr1)
        del x2, mask_output_curr_1c_1, mask_output_curr1

        state_curr2 = self.tail_state1(state_curr1.detach())
        mask_output_curr_1c_2 = self.tail_mask1(state_curr2.detach())
        mask_output_curr2 = torch.cat([mask_output_curr_1c_2, mask_output_curr_1c_2, mask_output_curr_1c_2], dim=1)
        masks.append(mask_output_curr2)
        del state_curr1, mask_output_curr_1c_2, mask_output_curr2

        state_curr3 = self.tail_state1(state_curr2.detach())
        mask_output_curr_1c_3 = self.tail_mask1(state_curr3.detach())
        mask_output_curr3 = torch.cat([mask_output_curr_1c_3, mask_output_curr_1c_3, mask_output_curr_1c_3], dim=1)
        masks.append(mask_output_curr3)
        del state_curr2, mask_output_curr_1c_3, mask_output_curr3

        state_curr4 = self.tail_state1(state_curr3.detach())
        mask_output_curr_1c_4 = self.tail_mask1(state_curr4.detach())
        mask_output_curr4 = torch.cat([mask_output_curr_1c_4, mask_output_curr_1c_4, mask_output_curr_1c_4], dim=1)
        masks.append(mask_output_curr4)
        del state_curr3, mask_output_curr_1c_4, mask_output_curr4
        #
        # state_curr5 = self.tail_state1(state_curr4.detach())
        # mask_output_curr_1c_5 = self.tail_mask1(state_curr5.detach())
        # mask_output_curr5 = torch.cat([mask_output_curr_1c_5, mask_output_curr_1c_5, mask_output_curr_1c_5], dim=1)
        # masks.append(mask_output_curr5)
        # del state_curr4, mask_output_curr_1c_5, mask_output_curr5
        #
        # state_curr6 = self.tail_state1(state_curr5.detach())
        # mask_output_curr_1c_6 = self.tail_mask1(state_curr6.detach())
        # mask_output_curr6 = torch.cat([mask_output_curr_1c_6, mask_output_curr_1c_6, mask_output_curr_1c_6], dim=1)
        # masks.append(mask_output_curr6)
        # del state_curr5, mask_output_curr_1c_6, mask_output_curr6
        #
        # state_curr7 = self.tail_state1(state_curr6.detach())
        # mask_output_curr_1c_7 = self.tail_mask1(state_curr7.detach())
        # mask_output_curr7 = torch.cat([mask_output_curr_1c_7, mask_output_curr_1c_7, mask_output_curr_1c_7], dim=1)
        # masks.append(mask_output_curr7)
        # del state_curr6, mask_output_curr_1c_7, mask_output_curr7
        #
        # state_curr8 = self.tail_state1(state_curr7.detach())
        # mask_output_curr_1c_8 = self.tail_mask1(state_curr8.detach())
        # mask_output_curr8 = torch.cat([mask_output_curr_1c_8, mask_output_curr_1c_8, mask_output_curr_1c_8], dim=1)
        # masks.append(mask_output_curr8)
        # del state_curr7, mask_output_curr_1c_8, mask_output_curr8
        #
        # state_curr9 = self.tail_state1(state_curr8.detach())
        # mask_output_curr_1c_9 = self.tail_mask1(state_curr9.detach())
        # mask_output_curr9 = torch.cat([mask_output_curr_1c_9, mask_output_curr_1c_9, mask_output_curr_1c_9], dim=1)
        # masks.append(mask_output_curr9)
        # del state_curr8, mask_output_curr_1c_9, mask_output_curr9
        #
        # state_curr10 = self.tail_state1(state_curr9.detach())
        # mask_output_curr_1c_10 = self.tail_mask1(state_curr10.detach())
        # mask_output_curr10 = torch.cat([mask_output_curr_1c_10, mask_output_curr_1c_10, mask_output_curr_1c_10], dim=1)
        # masks.append(mask_output_curr10)
        # del state_curr9, mask_output_curr_1c_10, mask_output_curr10
        #
        # state_curr11 = self.tail_state1(state_curr10.detach())
        # mask_output_curr_1c_11 = self.tail_mask1(state_curr11.detach())
        # mask_output_curr11 = torch.cat([mask_output_curr_1c_11, mask_output_curr_1c_11, mask_output_curr_1c_11], dim=1)
        # masks.append(mask_output_curr11)
        # del state_curr10, mask_output_curr_1c_11, mask_output_curr11
        #
        # state_curr12 = self.tail_state1(state_curr11.detach())
        # mask_output_curr_1c_12 = self.tail_mask1(state_curr12.detach())
        # mask_output_curr12 = torch.cat([mask_output_curr_1c_12, mask_output_curr_1c_12, mask_output_curr_1c_12], dim=1)
        # masks.append(mask_output_curr12)
        # del state_curr11, mask_output_curr_1c_12, mask_output_curr12
        #
        # state_curr13 = self.tail_state1(state_curr12.detach())
        # mask_output_curr_1c_13 = self.tail_mask1(state_curr13.detach())
        # mask_output_curr13 = torch.cat([mask_output_curr_1c_13, mask_output_curr_1c_13, mask_output_curr_1c_13], dim=1)
        # masks.append(mask_output_curr13)
        # del state_curr12, mask_output_curr_1c_13, mask_output_curr13
        #
        # state_curr14 = self.tail_state1(state_curr13.detach())
        # mask_output_curr_1c_14 = self.tail_mask1(state_curr14.detach())
        # mask_output_curr14 = torch.cat([mask_output_curr_1c_14, mask_output_curr_1c_14, mask_output_curr_1c_14], dim=1)
        # masks.append(mask_output_curr14)
        # del state_curr13, mask_output_curr_1c_14, mask_output_curr14
        #
        # state_curr15 = self.tail_state1(state_curr14.detach())
        # mask_output_curr_1c_15 = self.tail_mask1(state_curr15.detach())
        # mask_output_curr15 = torch.cat([mask_output_curr_1c_15, mask_output_curr_1c_15, mask_output_curr_1c_15], dim=1)
        # masks.append(mask_output_curr15)
        # del state_curr14, mask_output_curr_1c_15, mask_output_curr15
        #
        # state_curr16 = self.tail_state1(state_curr15.detach())
        # mask_output_curr_1c_16 = self.tail_mask1(state_curr16.detach())
        # mask_output_curr16 = torch.cat([mask_output_curr_1c_16, mask_output_curr_1c_16, mask_output_curr_1c_16], dim=1)
        # masks.append(mask_output_curr16)
        # del state_curr15, mask_output_curr_1c_16, mask_output_curr16
        #
        # state_curr17 = self.tail_state1(state_curr16.detach())
        # mask_output_curr_1c_17 = self.tail_mask1(state_curr17.detach())
        # mask_output_curr17 = torch.cat([mask_output_curr_1c_17, mask_output_curr_1c_17, mask_output_curr_1c_17], dim=1)
        # masks.append(mask_output_curr17)
        # del state_curr16, mask_output_curr_1c_17, mask_output_curr17
        #
        # state_curr18 = self.tail_state1(state_curr17.detach())
        # mask_output_curr_1c_18 = self.tail_mask1(state_curr18.detach())
        # mask_output_curr18 = torch.cat([mask_output_curr_1c_18, mask_output_curr_1c_18, mask_output_curr_1c_18], dim=1)
        # masks.append(mask_output_curr18)
        # del state_curr17, mask_output_curr_1c_18, mask_output_curr18
        #
        # state_curr19 = self.tail_state1(state_curr18.detach())
        # mask_output_curr_1c_19 = self.tail_mask1(state_curr19.detach())
        # mask_output_curr19 = torch.cat([mask_output_curr_1c_19, mask_output_curr_1c_19, mask_output_curr_1c_19], dim=1)
        # masks.append(mask_output_curr19)
        # del state_curr18, mask_output_curr_1c_19, mask_output_curr19
        #
        # state_curr20 = self.tail_state1(state_curr19.detach())
        # mask_output_curr_1c_20 = self.tail_mask1(state_curr20.detach())
        # mask_output_curr20 = torch.cat([mask_output_curr_1c_20, mask_output_curr_1c_20, mask_output_curr_1c_20], dim=1)
        # masks.append(mask_output_curr20)
        # del state_curr19, mask_output_curr_1c_20, mask_output_curr20
        #
        # state_curr21 = self.tail_state1(state_curr20.detach())
        # mask_output_curr_1c_21 = self.tail_mask1(state_curr21.detach())
        # mask_output_curr21 = torch.cat([mask_output_curr_1c_21, mask_output_curr_1c_21, mask_output_curr_1c_21], dim=1)
        # masks.append(mask_output_curr21)
        # del state_curr20, mask_output_curr_1c_21, mask_output_curr21
        #
        # state_curr22 = self.tail_state1(state_curr21.detach())
        # mask_output_curr_1c_22 = self.tail_mask1(state_curr22.detach())
        # mask_output_curr22 = torch.cat([mask_output_curr_1c_22, mask_output_curr_1c_22, mask_output_curr_1c_22], dim=1)
        # masks.append(mask_output_curr22)
        # del state_curr21, mask_output_curr_1c_22, mask_output_curr22
        #
        # state_curr23 = self.tail_state1(state_curr22.detach())
        # mask_output_curr_1c_23 = self.tail_mask1(state_curr23.detach())
        # mask_output_curr23 = torch.cat([mask_output_curr_1c_23, mask_output_curr_1c_23, mask_output_curr_1c_23], dim=1)
        # masks.append(mask_output_curr23)
        # del state_curr22, mask_output_curr_1c_23, mask_output_curr23
        #
        # state_curr24 = self.tail_state1(state_curr23.detach())
        # mask_output_curr_1c_24 = self.tail_mask1(state_curr24.detach())
        # mask_output_curr24 = torch.cat([mask_output_curr_1c_24, mask_output_curr_1c_24, mask_output_curr_1c_24], dim=1)
        # masks.append(mask_output_curr24)
        # del state_curr23, mask_output_curr_1c_24, mask_output_curr24
        #
        # state_curr25 = self.tail_state1(state_curr24.detach())
        # mask_output_curr_1c_25 = self.tail_mask1(state_curr25.detach())
        # mask_output_curr25 = torch.cat([mask_output_curr_1c_25, mask_output_curr_1c_25, mask_output_curr_1c_25], dim=1)
        # masks.append(mask_output_curr25)
        # del state_curr24, mask_output_curr_1c_25, mask_output_curr25
        #
        # state_curr26 = self.tail_state1(state_curr25.detach())
        # mask_output_curr_1c_26 = self.tail_mask1(state_curr26.detach())
        # mask_output_curr26 = torch.cat([mask_output_curr_1c_26, mask_output_curr_1c_26, mask_output_curr_1c_26], dim=1)
        # masks.append(mask_output_curr26)
        # del state_curr25, mask_output_curr_1c_26, mask_output_curr26
        #
        # state_curr27 = self.tail_state1(state_curr26.detach())
        # mask_output_curr_1c_27 = self.tail_mask1(state_curr27.detach())
        # mask_output_curr27 = torch.cat([mask_output_curr_1c_27, mask_output_curr_1c_27, mask_output_curr_1c_27], dim=1)
        # masks.append(mask_output_curr27)
        # del state_curr26, mask_output_curr_1c_27, mask_output_curr27
        #
        # state_curr28 = self.tail_state1(state_curr27.detach())
        # mask_output_curr_1c_28 = self.tail_mask1(state_curr28.detach())
        # mask_output_curr28 = torch.cat([mask_output_curr_1c_28, mask_output_curr_1c_28, mask_output_curr_1c_28], dim=1)
        # masks.append(mask_output_curr28)
        # del state_curr27, mask_output_curr_1c_28, mask_output_curr28
        #
        # state_curr29 = self.tail_state1(state_curr28.detach())
        # mask_output_curr_1c_29 = self.tail_mask1(state_curr29.detach())
        # mask_output_curr29 = torch.cat([mask_output_curr_1c_29, mask_output_curr_1c_29, mask_output_curr_1c_29], dim=1)
        # masks.append(mask_output_curr29)
        # del state_curr28, mask_output_curr_1c_29, mask_output_curr29
        #
        # state_curr30 = self.tail_state1(state_curr29.detach())
        # mask_output_curr_1c_30 = self.tail_mask1(state_curr30.detach())
        # mask_output_curr30 = torch.cat([mask_output_curr_1c_30, mask_output_curr_1c_30, mask_output_curr_1c_30], dim=1)
        # masks.append(mask_output_curr30)
        # del state_curr29, mask_output_curr_1c_30, mask_output_curr30
        #
        # state_curr31 = self.tail_state1(state_curr30.detach())
        # mask_output_curr_1c_31 = self.tail_mask1(state_curr31.detach())
        # mask_output_curr31 = torch.cat([mask_output_curr_1c_31, mask_output_curr_1c_31, mask_output_curr_1c_31], dim=1)
        # masks.append(mask_output_curr31)
        # del state_curr30, mask_output_curr_1c_31, mask_output_curr31
        #
        # state_curr32 = self.tail_state1(state_curr31.detach())
        # mask_output_curr_1c_32 = self.tail_mask1(state_curr32.detach())
        # mask_output_curr32 = torch.cat([mask_output_curr_1c_32, mask_output_curr_1c_32, mask_output_curr_1c_32], dim=1)
        # masks.append(mask_output_curr32)
        # del state_curr31, mask_output_curr_1c_32, mask_output_curr32

        # del mask_output_curr1, mask_output_curr2, mask_output_curr3, mask_output_curr4
        #
        # state_curr5 = self.tail_state1(state_curr4)
        # state_curr6 = self.tail_state1(state_curr5)
        # state_curr7 = self.tail_state1(state_curr6)
        # state_curr8 = self.tail_state1(state_curr7)
        #
        #
        # mask_output_curr_1c_5 = self.tail_mask1(state_curr5)
        # mask_output_curr_1c_6 = self.tail_mask1(state_curr6)
        # mask_output_curr_1c_7 = self.tail_mask1(state_curr7)
        # mask_output_curr_1c_8 = self.tail_mask1(state_curr8)
        #
        #
        # mask_output_curr5 = torch.cat([mask_output_curr_1c_5, mask_output_curr_1c_5, mask_output_curr_1c_5], dim=1)
        # mask_output_curr6 = torch.cat([mask_output_curr_1c_6, mask_output_curr_1c_6, mask_output_curr_1c_6], dim=1)
        # mask_output_curr7 = torch.cat([mask_output_curr_1c_7, mask_output_curr_1c_7, mask_output_curr_1c_7], dim=1)
        # mask_output_curr8 = torch.cat([mask_output_curr_1c_8, mask_output_curr_1c_8, mask_output_curr_1c_8], dim=1)
        #
        # masks.append(mask_output_curr5)
        # masks.append(mask_output_curr6)
        # masks.append(mask_output_curr7)
        # masks.append(mask_output_curr8)
        #
        # del mask_output_curr5, mask_output_curr6, mask_output_curr7, mask_output_curr8
        #
        # state_curr9 = self.tail_state1(state_curr8)
        # state_curr10 = self.tail_state1(state_curr9)
        # state_curr11 = self.tail_state1(state_curr10)
        # state_curr12 = self.tail_state1(state_curr11)
        #
        #
        # mask_output_curr_1c_9 = self.tail_mask1(state_curr9)
        # mask_output_curr_1c_10 = self.tail_mask1(state_curr10)
        # mask_output_curr_1c_11 = self.tail_mask1(state_curr11)
        # mask_output_curr_1c_12 = self.tail_mask1(state_curr12)
        #
        #
        # mask_output_curr9 = torch.cat([mask_output_curr_1c_9, mask_output_curr_1c_9, mask_output_curr_1c_9], dim=1)
        # mask_output_curr10 = torch.cat([mask_output_curr_1c_10, mask_output_curr_1c_10, mask_output_curr_1c_10], dim=1)
        # mask_output_curr11 = torch.cat([mask_output_curr_1c_11, mask_output_curr_1c_11, mask_output_curr_1c_11], dim=1)
        # mask_output_curr12 = torch.cat([mask_output_curr_1c_12, mask_output_curr_1c_12, mask_output_curr_1c_12], dim=1)
        #
        #
        #
        # masks.append(mask_output_curr9)
        # masks.append(mask_output_curr10)
        # masks.append(mask_output_curr11)
        # masks.append(mask_output_curr12)
        #
        # del mask_output_curr9, mask_output_curr10, mask_output_curr11, mask_output_curr12
        #
        # state_curr13 = self.tail_state1(state_curr12)
        # state_curr14 = self.tail_state1(state_curr13)
        # state_curr15 = self.tail_state1(state_curr14)
        # state_curr16 = self.tail_state1(state_curr15)
        #
        # mask_output_curr_1c_13 = self.tail_mask1(state_curr13)
        # mask_output_curr_1c_14 = self.tail_mask1(state_curr14)
        # mask_output_curr_1c_15 = self.tail_mask1(state_curr15)
        # mask_output_curr_1c_16 = self.tail_mask1(state_curr16)
        #
        # mask_output_curr13 = torch.cat([mask_output_curr_1c_13, mask_output_curr_1c_13, mask_output_curr_1c_13], dim=1)
        # mask_output_curr14 = torch.cat([mask_output_curr_1c_14, mask_output_curr_1c_14, mask_output_curr_1c_14], dim=1)
        # mask_output_curr15 = torch.cat([mask_output_curr_1c_15, mask_output_curr_1c_15, mask_output_curr_1c_15], dim=1)
        # mask_output_curr16 = torch.cat([mask_output_curr_1c_16, mask_output_curr_1c_16, mask_output_curr_1c_16], dim=1)
        #
        # masks.append(mask_output_curr13)
        # masks.append(mask_output_curr14)
        # masks.append(mask_output_curr15)
        # masks.append(mask_output_curr16)

        return torch.cat(masks,dim=1)

# class GeneratorResNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, res_blocks=9):
#         super(GeneratorResNet, self).__init__()
#         self.is_cuda = torch.cuda.is_available()
#         self.nfc = 32
#         self.min_nfc = 32
#         self.num_layer = 6
#         N = self.nfc
#
#         self.head = ConvBlock(in_channels, N, 3, 1, 1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
#         self.body = nn.Sequential()
#         mark = 0
#         for i in range(self.num_layer - 2):
#             block = ConvBlock(32, 32, 3, 1, 1)
#             self.body.add_module('block%d' % (i + 1), block)
#
#         self.tail_state1 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.tail_state2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.tail_state3 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.tail_state4 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.tail_mask1 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.tail_mask2 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.tail_mask3 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.tail_mask4 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(16),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )
#
#         self.tail_output = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             # MyConv2D(max(N,opt.min_nfc), opt.nc_im, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size),
#             nn.Tanh()
#         )
#
#         # self.tail_mask = nn.Sequential(
#         #     nn.Conv2d(out_channels * 2, 64, kernel_size=3, stride=1, padding=1),
#         #     nn.InstanceNorm2d(64),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#         #     nn.InstanceNorm2d(64),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
#         #     nn.Sigmoid()
#         # )
#
#         #for p in self.parameters():
#         #    p.requires_grad = False
#
#         #self.pretrained_dict = torch.load("saved_models/thirdstage/G_AB_250.pth")
#         # self.pretrained_dict = torch.load("saved_models/secondstage/G_AB_50.pth")
#         #
#         # self.model_dict = self.state_dict()
#         # self.model_dict.update(self.pretrained_dict)
#         # self.load_state_dict(self.model_dict)
#
#         for k,v in self.named_parameters():
#             print("k: {},  requires_grad: {}".format(k,v.requires_grad))
#
#     def forward(self, x):
#         input_curr = x
#         mask_input_curr = torch.ones(input_curr.shape).cuda()
#
#         outputs = []
#         masks = []
#         #for step in range(4):
#         x1 = self.head(input_curr)
#         x2 = self.body(x1)
#         state_curr1 = self.tail_state1(x2)
#         state_curr2 = self.tail_state1(state_curr1)
#         state_curr3 = self.tail_state1(state_curr2)
#         state_curr4 = self.tail_state1(state_curr3)
#         state_curr5 = self.tail_state1(state_curr4)
#         state_curr6 = self.tail_state1(state_curr5)
#         state_curr7 = self.tail_state1(state_curr6)
#         state_curr8 = self.tail_state1(state_curr7)
#         state_curr9 = self.tail_state1(state_curr8)
#         state_curr10 = self.tail_state1(state_curr9)
#         state_curr11 = self.tail_state1(state_curr10)
#         state_curr12 = self.tail_state1(state_curr11)
#         state_curr13 = self.tail_state1(state_curr12)
#         state_curr14 = self.tail_state1(state_curr13)
#         state_curr15 = self.tail_state1(state_curr14)
#         state_curr16 = self.tail_state1(state_curr15)
#
#         mask_output_curr_1c_1 = self.tail_mask1(state_curr1)
#         mask_output_curr_1c_2 = self.tail_mask1(state_curr2)
#         mask_output_curr_1c_3 = self.tail_mask1(state_curr3)
#         mask_output_curr_1c_4 = self.tail_mask1(state_curr4)
#         mask_output_curr_1c_5 = self.tail_mask1(state_curr5)
#         mask_output_curr_1c_6 = self.tail_mask1(state_curr6)
#         mask_output_curr_1c_7 = self.tail_mask1(state_curr7)
#         mask_output_curr_1c_8 = self.tail_mask1(state_curr8)
#         mask_output_curr_1c_9 = self.tail_mask1(state_curr9)
#         mask_output_curr_1c_10 = self.tail_mask1(state_curr10)
#         mask_output_curr_1c_11 = self.tail_mask1(state_curr11)
#         mask_output_curr_1c_12 = self.tail_mask1(state_curr12)
#         mask_output_curr_1c_13 = self.tail_mask1(state_curr13)
#         mask_output_curr_1c_14 = self.tail_mask1(state_curr14)
#         mask_output_curr_1c_15 = self.tail_mask1(state_curr15)
#         mask_output_curr_1c_16 = self.tail_mask1(state_curr16)
#
#         mask_output_curr1 = torch.cat([mask_output_curr_1c_1, mask_output_curr_1c_1, mask_output_curr_1c_1], dim=1)
#         mask_output_curr2 = torch.cat([mask_output_curr_1c_2, mask_output_curr_1c_2, mask_output_curr_1c_2], dim=1)
#         mask_output_curr3 = torch.cat([mask_output_curr_1c_3, mask_output_curr_1c_3, mask_output_curr_1c_3], dim=1)
#         mask_output_curr4 = torch.cat([mask_output_curr_1c_4, mask_output_curr_1c_4, mask_output_curr_1c_4], dim=1)
#         mask_output_curr5 = torch.cat([mask_output_curr_1c_5, mask_output_curr_1c_5, mask_output_curr_1c_5], dim=1)
#         mask_output_curr6 = torch.cat([mask_output_curr_1c_6, mask_output_curr_1c_6, mask_output_curr_1c_6], dim=1)
#         mask_output_curr7 = torch.cat([mask_output_curr_1c_7, mask_output_curr_1c_7, mask_output_curr_1c_7], dim=1)
#         mask_output_curr8 = torch.cat([mask_output_curr_1c_8, mask_output_curr_1c_8, mask_output_curr_1c_8], dim=1)
#         mask_output_curr9 = torch.cat([mask_output_curr_1c_9, mask_output_curr_1c_9, mask_output_curr_1c_9], dim=1)
#         mask_output_curr10 = torch.cat([mask_output_curr_1c_10, mask_output_curr_1c_10, mask_output_curr_1c_10], dim=1)
#         mask_output_curr11 = torch.cat([mask_output_curr_1c_11, mask_output_curr_1c_11, mask_output_curr_1c_11], dim=1)
#         mask_output_curr12 = torch.cat([mask_output_curr_1c_12, mask_output_curr_1c_12, mask_output_curr_1c_12], dim=1)
#         mask_output_curr13 = torch.cat([mask_output_curr_1c_13, mask_output_curr_1c_13, mask_output_curr_1c_13], dim=1)
#         mask_output_curr14 = torch.cat([mask_output_curr_1c_14, mask_output_curr_1c_14, mask_output_curr_1c_14], dim=1)
#         mask_output_curr15 = torch.cat([mask_output_curr_1c_15, mask_output_curr_1c_15, mask_output_curr_1c_15], dim=1)
#         mask_output_curr16 = torch.cat([mask_output_curr_1c_16, mask_output_curr_1c_16, mask_output_curr_1c_16], dim=1)
#
#         masks.append(mask_output_curr1)
#         masks.append(mask_output_curr2)
#         masks.append(mask_output_curr3)
#         masks.append(mask_output_curr4)
#         masks.append(mask_output_curr5)
#         masks.append(mask_output_curr6)
#         masks.append(mask_output_curr7)
#         masks.append(mask_output_curr8)
#         masks.append(mask_output_curr9)
#         masks.append(mask_output_curr10)
#         masks.append(mask_output_curr11)
#         masks.append(mask_output_curr12)
#         masks.append(mask_output_curr13)
#         masks.append(mask_output_curr14)
#         masks.append(mask_output_curr15)
#         masks.append(mask_output_curr16)
#
#         return torch.cat(masks,dim=1)

##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

import torch.nn as nn
import torch.nn.functional as F
import torch
import time

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

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
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

class BasicBlock(nn.Module):
    def __init__(self, type, inplane, outplane, stride):
        super(BasicBlock, self).__init__()
        conv_block = []
        if type == "Conv":
            conv_block += [nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1)]
        elif type == "Deconv":
            conv_block += [nn.ConvTranspose2d(inplane, outplane, kernel_size=4, stride=stride, padding=1)]

        conv_block +=[nn.InstanceNorm2d(outplane),
                      nn.ReLU()]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, res_blocks=6):
        super(GeneratorResNet, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.nfc = 32
        self.min_nfc = 32
        self.num_layer = 8
        N = self.nfc

        model = []

        model += [BasicBlock("Conv", 6, 64, 1)]
        model += [BasicBlock("Conv", 64, 64, 1)]
        model += [BasicBlock("Conv", 64, 64, 2)]

        model += [ResnetBlock(64, 2)]
        model += [ResnetBlock(64, 2)]
        model += [ResnetBlock(64, 4)]
        model += [ResnetBlock(64, 4)]
        model += [ResnetBlock(64, 8)]
        model += [ResnetBlock(64, 8)]
        model += [ResnetBlock(64, 16)]
        model += [ResnetBlock(64, 16)]
        model += [ResnetBlock(64, 1)]
        model += [ResnetBlock(64, 1)]

        model += [BasicBlock("Deconv", 64, 64, 2)]
        model += [BasicBlock("Conv", 64, 64, 1)]
        model += [nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*model)


        for k,v in self.named_parameters():
            print("k: {},  requires_grad: {}".format(k,v.requires_grad)) 

    def forward(self, x, edges, seed):
        input_curr = x
        #state_prev = torch.zeros(input_curr.shape).cuda()
        outputs = []
        if seed % 9 != 0:
            for step in range(4):
                residual_curr = self.model(torch.cat([input_curr, edges[:,step*3:step*3+3,:,:]], dim=1))
                output_curr = input_curr + residual_curr.detach()
                input_curr = output_curr.detach()
                outputs.append(output_curr)
                del residual_curr
        else:
            for step in range(4):

                residual_curr = self.model(torch.cat([x, edges[:,step*3:step*3+3,:,:]], dim=1))
                output_curr = x + residual_curr

                                                        
                outputs.append(output_curr)
 
        return torch.cat(outputs,dim=1)

##############################
#        Discriminator
##############################

#class Discriminator(nn.Module):
#    def __init__(self, in_channels=3):
#        super(Discriminator, self).__init__()
#
#        def discriminator_block(in_filters, out_filters, normalize=True):
#            """Returns downsampling layers of each discriminator block"""
#            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#            if normalize:
#                layers.append(nn.InstanceNorm2d(out_filters))
#            layers.append(nn.LeakyReLU(0.2, inplace=True))
#            return layers
#
#        self.model = nn.Sequential(
#            *discriminator_block(in_channels, 64, normalize=False),
#            *discriminator_block(64, 128),
#            *discriminator_block(128, 256),
#            *discriminator_block(256, 512),
#            nn.ZeroPad2d((1, 0, 1, 0)),
#            nn.Conv2d(512, 1, 4, padding=1)
#        )
#
#    def forward(self, img):
#        return self.model(img)

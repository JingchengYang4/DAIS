import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Edge_Occlusion(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        #self.normalize = cfg.MODEL.DEPTH.OBJECT_NORMALIZATION

        """self.fconv1 = nn.Conv2d(in_channels=257, out_channels=128, kernel_size=1)
        self.fconv2 = nn.Conv2d(in_channels=129, out_channels=64, kernel_size=1)
        self.fconv3 = nn.Conv2d(in_channels=65, out_channels=32, kernel_size=1)
        self.fconv4 = nn.Conv2d(in_channels=33, out_channels=16, kernel_size=1)"""

        self.mode = 1

        if self.mode is 1:
            self.edge_conv0 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1)
            self.edge_conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=2)#size of 7
            self.edge_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)#I BELIEVE OUPUT SIZE IS 3
            self.edge_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

            # (1-1) x stride + kernel = 3
            self.dconv0 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
            self.upconv0 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1)

            # (3-1) x stride + kernel = 7
            # 2 x 2 + 2 = 6
            self.dconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            self.upconv1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1)
            # (7-1) x stride + kernel = 14
            # 6 x 2 + 2 = 14
            self.dconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
            self.upconv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)

            self.fconv = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        elif self.mode is 2:
            self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.c2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.c3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
            self.c4 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
            self.c5 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        elif self.mode is 3:
            self.c1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
            self.c2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
            self.c3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        self.loss = nn.BCEWithLogitsLoss()
        self.vis_period = cfg.VIS_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        if self.vis_period > 0:
            self.i = 0
            os.makedirs(self.output_dir + '/visualization/', exist_ok=True)

    def forward(self, depth, gt, visible=None):

        x = depth

        if self.mode is 1:
            x = self.edge_conv0(x)
            x = F.relu(x)
            x0 = x
            x = self.edge_conv1(x)
            x = F.relu(x)
            x1 = x
            x = self.edge_conv2(x)
            x = F.relu(x)
            x2 = x
            x = self.edge_conv3(x)
            x = F.relu(x)

            x = self.dconv0(x)
            x = F.relu(x)

            x = self.upconv0(torch.cat((x, x2), 1))
            x = F.relu(x)

            x = self.dconv1(x)
            x = F.relu(x)

            x = self.upconv1(torch.cat((x, x1), 1))
            x = F.relu(x)

            x = self.dconv2(x)
            x = F.relu(x)

            x = self.upconv2(torch.cat((x, x0), 1))
            x = F.relu(x)

            x = self.fconv(x)
        elif self.mode is 2:
            x = self.c1(x)
            x = F.relu(x)
            x = self.c2(x)
            x = F.relu(x)
            x = self.c3(x)
            x = F.relu(x)
            x = self.c4(x)
            x = F.relu(x)
            x = self.c5(x)
        elif self.mode is 3:
            x = self.c1(x)
            x = F.relu(x)
            x = self.c2(x)
            x = F.relu(x)
            x = self.c3(x)

        if self.vis_period > 0:
            if self.i % 100 == 0 and x.size()[0] > 0:
                f, axarrr = plt.subplots(2, 1)
                axarrr[0].imshow(x[0][0].cpu().detach().numpy())
                axarrr[1].imshow(gt[0][0].cpu().detach().numpy())
                #plt.show()
                plt.savefig(self.output_dir + '/visualization/OE_' + str(self.i) + '.png', dpi=800)

        self.i += 1
        oe_loss = self.loss(x, gt)
        return oe_loss, x
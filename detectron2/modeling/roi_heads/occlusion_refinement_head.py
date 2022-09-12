import torch
import torch.nn as nn
import torch.nn.functional as F

class Occlusion_Refinement(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.c1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, amodal, oe, gt=None):

        x = torch.cat((amodal, oe), dim=1)
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c4(x)

        if gt is None:
            return torch.sigmoid(x)
        else:
            return self.loss(x, gt)
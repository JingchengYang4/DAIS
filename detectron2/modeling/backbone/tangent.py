import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt


class Tangent:
    def __init__(self, cfg):

        self.kernel_size = cfg.MODEL.DEPTH.SOBEL_KERNEL_SIZE
        self.kernel_range = int((self.kernel_size - 1)/2)

        self.dx_kernel = torch.zeros((self.kernel_size, self.kernel_size)).cuda()
        self.dy_kernel = torch.zeros((self.kernel_size, self.kernel_size)).cuda()

        for i in range(-self.kernel_range, self.kernel_range+1):
            #print(i)
            for j in range(-self.kernel_range, self.kernel_range+1):
                if i == 0 and j == 0:
                    #print("CONTINUE", i, j)
                    continue
                self.dx_kernel[i+self.kernel_range][j+self.kernel_range] = i / (i*i + j*j)
                self.dy_kernel[i+self.kernel_range][j+self.kernel_range] = j / (i*i + j*j)

        self.dx_kernel *= 2.0
        self.dy_kernel *= 2.0
        self.dx_kernel = self.dx_kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(1, 1, 1, 1)
        self.dy_kernel = self.dy_kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(1, 1, 1, 1)
        self.normal_strength = cfg.MODEL.DEPTH.NORMAL_MAP_STRENGTH
        self.visualize_normal = cfg.MODEL.DEPTH.VISUALIZE_NORMAL
        self.extract_normal = cfg.MODEL.DEPTH.EXTRACT_NORMAL

    def get_normals(self, depth):
        dx = F.conv2d(depth, self.dx_kernel, padding=self.kernel_range) * -1.0
        dy = F.conv2d(depth, self.dy_kernel, padding=self.kernel_range) * -1.0

        #print(torch.min(dx), torch.max(dx))

        dz = torch.full(depth.size(), self.normal_strength).cuda()

        tangent = torch.cat([dx, dy, dz], dim=1)
        magnitude = tangent.norm(dim=1, p=2)
        normal = tangent / magnitude
        del tangent
        del magnitude
        #print(tangent.size())

        if self.visualize_normal:
            dx = torch.abs(dx)*5
            dx = torch.sqrt(dx)

            dy = torch.abs(dy)*5
            dy = torch.sqrt(dy)

            #dx = torchvision.transforms.functional.gaussian_blur(dx, 10)

            f, axarrr = plt.subplots(3, 1)
            axarrr[0].imshow(dx.cpu()[0][0])
            axarrr[1].imshow(dy.cpu()[0][0])
            axarrr[2].imshow((normal/2+0.5).cpu()[0].permute(1, 2, 0))
            plt.show()

        return normal

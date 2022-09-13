import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from detectron2.modeling.backbone.depth_prediction import DepthPredictionModule
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

depth_predictor = DepthPredictionModule(width=352, height=1216)

file = open("depth_train.txt", "r")
lines = file.read().splitlines()

convert = transforms.Compose([transforms.ToTensor()])
i = 0
for file in lines:
    image = Image.open(file)
    image = convert(image).unsqueeze(0).cuda()
    depth = F.interpolate(image, size=(352, 1216), mode='bilinear', align_corners=True)
    depth = depth_predictor.Predict(depth)
    depth = F.interpolate(depth, image[0][0].size(), mode='bilinear', align_corners=True)
    #print(i, "/", len(lines))
    i += 1
    depth = torch.log10(depth)/3
    #print(torch.min(depth), torch.max(depth))
    save_image(depth[0], file.replace(".png", "_depth.png"))
    #print(file)
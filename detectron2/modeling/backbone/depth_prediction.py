import argparse

from torchvision import transforms

from depth.pytorch.bts import BtsModel
import torch


class DepthPredictionModule:
    def __init__(self, height=800, width=2656):
        parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
        args = parser.parse_args([])
        args.encoder = 'densenet161_bts'
        args.dataset = 'kitti'
        args.model_name = 'bts_eigen_v2_pytorch_densenet161'
        args.checkpoint_path = 'depth/pytorch/models/bts_eigen_v2_pytorch_densenet161/model'
        args.input_height = height
        args.input_width = width
        #args.input_height = 400
        #args.input_width = 1328
        args.max_depth = 80
        args.do_kb_crop = True
        args.bts_size = 512
        model = BtsModel(params=args)
        model = torch.nn.DataParallel(model)
#800x2656
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.cuda()
        self.model = model

    def Predict(self, x):
        with torch.no_grad():
            #normalization code based on bts
            image = x
            image = transforms.functional.normalize(image[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = image.unsqueeze(0)
            focal = torch.tensor([721])
            #focal is 721
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = self.model(image, focal)
            return depth_est
        #return depth_est

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.utils.visualizer import Visualizer

from ..backbone import build_backbone
from detectron2.modeling.backbone.depth_prediction import DepthPredictionModule
from ..backbone.tangent import Tangent
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

import torch.nn.functional as F

import os

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.input_format = cfg.INPUT.FORMAT
        self.vis_period = cfg.VIS_PERIOD
        self.predict_depth = cfg.MODEL.DEPTH.PREDICT
        self.output_dir = cfg.OUTPUT_DIR
        self.no_rgb = cfg.MODEL.NO_RGB
        if self.predict_depth:
            self.extract_depth = cfg.MODEL.DEPTH.EXTRACT_FEATURES
            self.depth_predictor = DepthPredictionModule()
            self.depth_std = cfg.MODEL.DEPTH.PIXEL_STD
            self.depth_mean = cfg.MODEL.DEPTH.PIXEL_MEAN
            self.visualize_depth = cfg.MODEL.DEPTH.VISUALIZE
            self.normal_map = cfg.MODEL.DEPTH.NORMAL_MAP
            self.normalize_depth = cfg.MODEL.DEPTH.NORMALIZE_DEPTH
            if self.normalize_depth:
                self.norm_step = cfg.MODEL.DEPTH.NORMALIZE_STEP
                self.offsets = cfg.MODEL.DEPTH.NORMALIZE_SUB
                self.visualize_depth_norm = cfg.MODEL.DEPTH.VISUALIZE_DEPTH_NORM
            if self.normal_map:
                self.tangent = Tangent(cfg)
                self.extract_normal = cfg.MODEL.DEPTH.EXTRACT_NORMAL


        if self.vis_period > 0:
            os.makedirs(self.output_dir + '/visualization/', exist_ok=True)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """

        inputs = [x for x in batched_inputs]
        prop_boxes = [p for p in proposals]
        storage = get_event_storage()
        max_vis_prop = 20
        index = 0
        for input, prop in zip(inputs, prop_boxes):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes, masks=input["instances"].gt_masks)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)
            #print(vis_img.shape)
            plt.imshow(np.transpose(vis_img, (1, 2, 0)))
            #plt.gca().set_aspect(1)
            plt.savefig(self.output_dir + '/visualization/' + str(storage.iter) + '_' + str(index) + '.png', dpi=500)
            index += 1
            plt.close()
            #plt.show()
            #print("OK", vis_img)

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
            do_postprocess: (bool): whether to apply post-processing on the outputs.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs, do_postprocess=do_postprocess)
        images = self.preprocess_image(batched_inputs)

        if False:
            print(batched_inputs[0]['instances'])
            gtmasks = batched_inputs[0]['instances'].gt_visible_masks.tensor
            for mask in gtmasks:
                plt.imshow(mask.numpy())
                plt.show()
            quit()

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.predict_depth:
            images.tensor, depths = self.depth(images.tensor)

        #print("STUFF I GUESS")

        #I WILL DEAL WITH YOU IN THE FUTURE, AGHHHH!!!
        features = self.backbone(images.tensor)
        if self.proposal_generator:
            #this is true when running the code
            #print(self.proposal_generator)
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        #print(self.roi_heads)
        #roi head shere
        if self.predict_depth:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, depth=depths[0])
        else:
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def depth(self, images):
        image = torch.flip(images, [1])#switches bgr to rgb
        image -= torch.min(image)#recenter RGB to 0-255
        image /= 255
        depth = self.depth_predictor.Predict(image)
        depth = np.log10(depth)
        depth_tensor = torch.tensor([[depth]]).cuda()
        depth_tensor = (depth_tensor - torch.mean(depth_tensor)) / torch.std(depth_tensor)

        #print(torch.min(depth_tensor), torch.max(depth_tensor))
        output = [depth_tensor]

        if self.normalize_depth:
            norm_depths = depth_tensor
            for i in range(1, self.offsets): #0, 1, 2, 3
                offset = i * (1/self.offsets) * self.norm_step
                norm_depths = torch.cat((norm_depths, depth_tensor+offset), 1)

            depth_norm = norm_depths/self.norm_step
            depth_norm = (depth_norm - (torch.sin(2 * 3.1415926 * depth_norm))/(2 * 3.1415926)) * self.norm_step

            norm_depths -= depth_norm

            if self.visualize_depth_norm:
                f, axarrr = plt.subplots(self.offsets, 1)
                for i in range(0, self.offsets):
                    axarrr[i].imshow(norm_depths[0][i].cpu())
                plt.show()

            output.append(norm_depths)

        if self.extract_depth:
            if self.normalize_depth:
                images = torch.cat((images, norm_depths), 1)
                #print(images.size())
            else:
                images = torch.cat((images, depth_tensor), 1)

        #print(torch.min(depth_tensor), torch.max(depth_tensor))

        if self.normal_map:
            normal = self.tangent.get_normals(depth_tensor)
            output.append(normal)

            if self.extract_normal:
                images = torch.cat((images, normal), 1)

        if self.visualize_depth:
            f, axarrr = plt.subplots(2, 1)
            axarrr[0].imshow(image.cpu()[0].permute(1, 2, 0))
            #print(depth, type(depth))
            axarrr[1].imshow(depth_tensor.cpu()[0][0])
            plt.show()
            #quit()

        if self.no_rgb:
            images = images[:, 3:, :, :]
        #print(images.size())
        return images, output

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)

        if self.predict_depth:
            images.tensor, output = self.depth(images.tensor)

        features = self.backbone(images.tensor)
        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            gt_instances = [x["inference_instances"].to(self.device) for x in batched_inputs]
            if self.predict_depth:
                results, _ = self.roi_heads(images, features, proposals, gt_instances, depth=output[0])
            else:
                results, _ = self.roi_heads(images, features, proposals, gt_instances)
            # results, _ = self.roi_heads(images, features, proposals, None)
            #so inference is here then
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

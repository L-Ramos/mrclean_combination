import os
import logging
import argparse
import numpy as np
from glob import glob
from typing import Tuple
import SimpleITK as sitk

import torch
from torch.utils.data import DataLoader

from unet_model import ResUNet_3
from dataset import InferenceDataset
from utils import read_image, load_model, save_image

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

def inference(model: ResUNet_3, dataset: InferenceDataset, batch_size, threshold) -> Tuple[sitk.Image, sitk.Image]:

    with torch.no_grad():

        # Loading dataset to the Dataloader to have multiple
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            #num_workers=1,
            pin_memory=USE_CUDA,
            drop_last=False
        )

        # Intialise the prediction and probability arrays
        pred = torch.Tensor(np.zeros(dataset.cta_np.shape)).to(device)
        prob = torch.Tensor(np.zeros(dataset.cta_np.shape)).to(device)

        for _, sample in enumerate(dataloader):

            for key, value in sample.items():
                sample[key] = value.to(device)

            pred_seg = model(sample['cta'])
            pred_seg = pred_seg.squeeze(1)

            tmp_pred = torch.zeros_like(pred_seg)
            tmp_pred[pred_seg > threshold] = 1
            pred[sample['idx']] = tmp_pred
            prob[sample['idx']] = pred_seg

        # Save the predictions
        pred_image = sitk.GetImageFromArray(pred.detach().cpu().numpy())
        pred_image = sitk.Cast(pred_image, sitk.sitkUInt8)
        pred_image.CopyInformation(dataset.r_cta)

        prob_image = sitk.GetImageFromArray(prob.detach().cpu().numpy())
        prob_image.CopyInformation(dataset.r_cta)

        return pred_image, prob_image


def segment_brain(batch_size: int, inp_cta: sitk.Image, model: ResUNet_3) -> sitk.Image:
    """ Segment brain from CTA scan.
    
    Args:
        batch_size (int): Number of slices to be segmented in parallel. 
        inp_cta (sitk.Image): Input image.
        model_file_path (str): Path to model parameters.
    
    Returns:
        sitk.Image: Returns the segmentation image
    """

    # Initialising the model
    #model = load_model(model_file_path, device)

    # Loading the dataset
    dataset = InferenceDataset(cta=inp_cta)

    # Run inference
    threshold = 0.5 # Fixed threshold
    pred_bm_image, _ = inference(model, dataset, batch_size, threshold)

    return pred_bm_image


def segment_vessels(batch_size: int, threshold: float, input_path_ct: str, output_dir: str, vessel_param_path: str, \
    bm_param_path: str) -> str:
    """ Segment vessels from CTA scan.

    Args:
        batch_size: Number of slices to be segmented in parallel 
        threshold: The probability threshold to make a binary decision as to 
         whether a voxel is vessel or not.
        input_path_ct: Path to input image.
        output_dir: Path to folder where to store the output.
        model_file_path: Path to the model parameters.

    Returns:
        str: Returns the path where the output is saved
    """
    # start_time = datetime.now()

    os.makedirs(output_dir, exist_ok=True)

    # Run Brain Segmentation
    cta = read_image(file_path=input_path_ct)
    bm = segment_brain(batch_size, cta, bm_param_path)


    # Initialising the model
    model = load_model(vessel_param_path, device)

    # Loading the dataset
    dataset = InferenceDataset(cta=cta, vesselseg=True, brain_mask=bm)

    # Run inference
    pred_vs_image, prob_vs_image = inference(model, dataset, batch_size, threshold)

    output_path = os.path.join(output_dir, 'segmentation.nii.gz')
    output_probability_path = os.path.join(output_dir, 'prob_segmentation.nii.gz')
    bm_path = os.path.join(output_dir, 'brain_mask.nii.gz')
    resize_image_path = os.path.join(output_dir, 'r_image.nii.gz')

    # save the outputs
    save_image(pred_vs_image, output_path) 
    save_image(prob_vs_image, output_probability_path)
    # save the intermediate results
    save_image(bm, bm_path)
    save_image(dataset.r_cta, resize_image_path)

    # print("Time taken:", datetime.now()-start_time)

    return output_path

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_path", type=str, help='Path to where the CTA image(s) are located.')
    arg_parser.add_argument("--output_path", type=str, default=None, 
                            help='Path to where the output will be stored.')
    arg_parser.add_argument("--threshold", type=float, default=0.5, 
                            help='Probability threshold to decide between vessel or not')
    arg_parser.add_argument("--vessel_model_path", type=str, default="vessel_param.tar",
                            help='Path to the vessel segmentation model parameters')
    arg_parser.add_argument("--bm_model_path", type=str, default="bm_param.tar",
                            help='Path to the brain segmentation model parameters')
    arg_parser.add_argument("--batch_size", type=int, default=4, 
                            help='Number of slices to be segmented in parallel')
    args = arg_parser.parse_args()


    if args.output_path == None:
        folder_path, _ = os.path.split(args.input_path)
        args.output_path = folder_path

    output_path = segment_vessels(
        batch_size = args.batch_size,
        threshold= args.threshold,
        input_path_ct= args.input_path,
        output_dir= args.output_path,
        vessel_param_path=args.vessel_model_path,
        bm_param_path=args.bm_model_path
    )




import os
import math
import torch
import logging
import numpy as np
import SimpleITK as sitk
from typing import List, Tuple
from skimage.morphology import dilation

from unet_model import ResUNet_3

def read_image(file_path: str) -> sitk.Image:

    if not os.path.isfile(file_path):
        raise IOError(f"Image is not found at path {file_path}")

    try:
        cta = sitk.ReadImage(file_path)
    except Exception as e:
        print(f"Unable to open the file from {file_path}")
        raise e

    return cta


def save_image(image: sitk.Image, output_path: str):

    try:
        sitk.WriteImage(image, output_path)
    except Exception as e:
        print(f'Unable to write the image to the given path, {output_path}')
        raise e

def load_model(model_file_path: str, device: torch.device):

    model = ResUNet_3()

    if not os.path.isfile(model_file_path):
        raise IOError(f'Model file not found at {model_file_path}')

    try:
        checkpoint = torch.load(model_file_path)
    except Exception as e:
        print(f"Unable to load the model from {model_file_path}")
        raise e

    if 'model_state_dict' not in checkpoint:
        raise IOError(f'The model file doesnt have the parameters stored under model_state_dict key')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    return model



def compute_spacing_same_physical_size(img: sitk.Image, new_size: Tuple[int, int, int]) -> Tuple[float]:
    """ Compute spacing required for a given new size in the same physical space.

    Args:
        img (sitk.Image): Simple ITK image of N dimensions.
        new_size (tuple of int): Size of the new image. Tuple of length N.

    Returns:
        The spacing.
    """
    new_size = np.asarray(new_size)
    return physical_size(img) / new_size


def physical_size(img: sitk.Image) -> List[float]:
    """ Compute physical size of image.

    Args:
        img: Simple ITK image of N dimensions.

    Returns:
        (list of float): Physical length of the N dimensions.

    """
    return [space * voxels for space, voxels in zip(img.GetSpacing(), img.GetSize())]


def resize(image: sitk.Image, interpolator: int, new_size: Tuple[int, int, int]) -> sitk.Image:
    """
    Args:
        image (sitk.Image): The input image.
        interpolator (int): Interpolator for the image
            (sitk.sitkNearestNeighbor) -> 1:  for binary image
            (sitk.sitkLinear) -> 2:  for float image
        new_size (tuple): The image will be resampled to this new size.

    Returns:
        sitk.Image: Resampled image.
    """
    assert isinstance(image, sitk.Image), "ct must be a sitk.Image type"
    assert interpolator == 2 or interpolator == 1, "Interpolator must be sitk.sitkNearestNeighbor (1)\
        or sitk.sitkLinear (2)"
    assert isinstance(new_size, tuple), "new_size should be a tuple"
    assert len(
        new_size) == 3, f"The dimensions of new size should be 3. But the input had dimension of {len(new_size)}"

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)

    new_spacing = compute_spacing_same_physical_size(image, new_size)
    resample_filter.SetOutputSpacing(new_spacing)

    resample_filter.SetOutputPixelType(image.GetPixelID())
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())

    return resample_filter.Execute(image)


def unit_scale_array(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """ Scale numpy array between 0 and 1.

    Args:
        arr: Input array of any size.

    Returns:
        Rescaled input of type float, original max and min value (for descaling)
    """
    max_value = arr.max()
    min_value = arr.min()

    range_value = max_value - min_value
    assert range_value > 0, "The range of the image must be greater than 0"

    arr_out = (arr - min_value) / range_value
    return arr_out, max_value, min_value


def normalise(img: sitk.Image) -> sitk.Image:
    """
    Args:
        img (sitk.Image): Input image

    Returns:
        sitk.Image: Normalised image.
    """
    assert isinstance(img, sitk.Image), "ct must be a sitk.Image type"

    ct_np = sitk.GetArrayFromImage(img)
    ct_np = ct_np.astype(np.int32)

    ct_nm, _, _ = unit_scale_array(ct_np)

    ct_nm_image = sitk.GetImageFromArray(ct_nm)
    ct_nm_image.CopyInformation(img)

    return ct_nm_image


def get_brain(cta: sitk.Image, bm: sitk.Image) -> sitk.Image:
    """Returns only the brain tissue given the image and the brain mask.
    
    Args:
        cta (sitk.Image): The input CTA image
        bm (sitk.Image): The brain segmentation image
    
    Returns:
        sitk.Image : Image with only the brain tissue.
    """
    cta_np = sitk.GetArrayFromImage(cta)
    bm_np = sitk.GetArrayFromImage(bm)

    d_bm_np = dilation(bm_np, np.ones((20, 20, 20)))

    b_cta_np = d_bm_np*cta_np

    b_cta = sitk.GetImageFromArray(b_cta_np)
    b_cta.CopyInformation(cta)

    return b_cta

import os
import numpy as np
import SimpleITK as sitk
import logging
from typing import Dict, Union

from torch.utils.data import Dataset

from utils import resize, normalise, get_brain, save_image


class InferenceDataset(Dataset):
    """This class loads the image and performs pre-processing required for model.
     Using the Dataset class to speed up the inference time with batching.
    """
    def __init__(self, cta: sitk.Image, vesselseg: bool = False, brain_mask=None ) -> None:

        self.logger = logging.getLogger(__name__)

        self.cta = cta

        # resampling to (x, 512, 512) if required
        image_size = self.cta.GetSize()
        if image_size[0] != 512 or image_size[1] != 512:
            new_size = (512, 512, image_size[2])
            self.r_cta = resize(self.cta, sitk.sitkLinear, new_size)
            self.logger.info(f'Original image size is {image_size}. Resizing the image to {new_size}.')
        else:
            self.r_cta = self.cta

        if vesselseg:
            #perform brain removal
            inp_nr_cta = get_brain(self.r_cta, brain_mask)
        else:
            inp_nr_cta = self.r_cta

        # normalising the image values to 0-1
        self.nr_cta = normalise(inp_nr_cta)
        self.logger.info('Image is normalised.')

        self.cta_np = sitk.GetArrayFromImage(self.nr_cta)
        self.cta_np = self.cta_np.astype(np.float32)

        self.num_slices = self.cta_np.shape[0]
        self.logger.info(f'Image is loaded. It has {self.num_slices} slices.')

    def __len__(self) -> int:
        
        return self.num_slices-2

    def __getitem__(self, idx: str) -> Dict[str, Union[np.ndarray, str]]:

        idx = int(idx) + 1

        _data = dict()
        _data['cta'] = self.cta_np[idx-1:idx+2, :, :]
        _data['idx'] = idx

        return _data

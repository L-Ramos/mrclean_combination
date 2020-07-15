 #TODO Use better naming for intermediate variables.
import SimpleITK as sitk
import math
import numpy as np

def Segment_Brain(image,foramen_radius = 7):
    """
    image          -- input image
    min_brain_HU   -- minimum value of Hounsfield unit that is considered as brain tissue
    max_brain_HU   -- maximum value of Hounsfield unit that is considered as brain tissue
    min_skull_HU   -- bone tissue is everything with this intensity value and above
    foramen_radius -- radius (in mm) of biggest hole in the skull after the foramen magnum
    
    For CTA scans, standard values are the following:
    min_skull_HU = 350
    min_brain_HU = -20
    max_brain_HU = 330

    For non-constrast CT scans, standard values are the following:
    min_skull_HU = 160
    min_brain_HU = -20
    max_brain_HU = 140
    """
    input_image = sitk.Cast(image, sitk.sitkInt32)

    min_skull_HU = 350
    min_brain_HU = -20
    max_brain_HU = 330
    
#    min_skull_HU = 160
#    min_brain_HU = -20
#    max_brain_HU = 140

    foramen_radius_3d = (math.floor(foramen_radius / input_image.GetSpacing()[0]),
                         math.floor(foramen_radius / input_image.GetSpacing()[1]),
                         math.floor(foramen_radius / input_image.GetSpacing()[2]))

    # Segment all bone in the image.
    aux_image = sitk.BinaryThreshold(input_image, min_skull_HU, 65535)

    # Close skull holes
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetBackgroundValue(0)
    dilate.SetForegroundValue(1)
    dilate.SetKernelRadius(foramen_radius_3d)
    aux_image = dilate.Execute(aux_image)
    skull_without_holes = aux_image

    #sitk.WriteImage(skull_without_holes, "skull_without_holes.mha")

    # For each scan slice, segment area outside the skull.
    slice_id = input_image.GetSize()[2] - 1
    while slice_id >= 0:
        aux_slice = aux_image[:, :, slice_id]
        region_growing_filter = sitk.ConnectedThresholdImageFilter()
        region_growing_filter.SetLower(0)
        region_growing_filter.SetUpper(0)
        region_growing_filter.SetSeed([0, 0])
        slice_mask = region_growing_filter.Execute(aux_slice)
        slice_mask = sitk.JoinSeries(slice_mask)
        aux_image = sitk.Paste(aux_image, slice_mask, slice_mask.GetSize(), destinationIndex = [0, 0, slice_id])
        slice_id -= 1

    # Dilate the segmentation of the area outside the skull back to the skull border.
    outside_skull_mask = dilate.Execute(aux_image)

    #sitk.WriteImage(outside_skull_mask, "D:/MRCLEAN/REGISTRY/CTA_THIN/R2061/outside_skull_mask.mha")

    # Remove other connected components that are not part of the brain.
    outside_brain_mask = outside_skull_mask + skull_without_holes
    outside_brain_mask = sitk.Clamp(outside_brain_mask, sitk.sitkInt32, lowerBound = 0, upperBound = 1)
    outside_brain_mask = sitk.InvertIntensity(outside_brain_mask, 1)
    outside_brain_mask = sitk.RelabelComponent(sitk.ConnectedComponent(outside_brain_mask))
    outside_brain_mask = sitk.Threshold(outside_brain_mask)

    # Dilate the segmentation of the area inside the skull back to the skull border.
    outside_brain_mask = dilate.Execute(outside_brain_mask)
    outside_brain_mask = sitk.InvertIntensity(outside_brain_mask, 1)
    outside_brain_mask = sitk.Cast(outside_brain_mask, sitk.sitkInt32)

    #sitk.WriteImage(outside_brain_mask, "D:/MRCLEAN/REGISTRY/CTA_THIN/R2061/outside_brain_mask.mha")

    # Detect neck base by evaluating slice's segmented area and connected components (1 component and area < 900mm²).
    neck_slice = 0
    slice_id = 0
    while slice_id < input_image.GetSize()[2]:
        slice_mask = outside_brain_mask[:, :, slice_id]
        slice_mask = sitk.ConnectedComponent(slice_mask)
        label_info_filter = sitk.LabelStatisticsImageFilter()
        label_info_filter.Execute(slice_mask, slice_mask)
        area = label_info_filter.GetCount(0) * input_image.GetSpacing()[0] * input_image.GetSpacing()[1]
        components = label_info_filter.GetNumberOfLabels()
        if components > 2 or area > 900:
            break
        neck_slice = slice_id
        slice_id += 1
    slice_id = neck_slice
    while slice_id >= 0:
        slice_mask = outside_brain_mask[:, :, slice_id]
        slice_mask = (slice_mask * 0) + 1
        slice_mask = sitk.JoinSeries(slice_mask)
        outside_brain_mask = sitk.Paste(outside_brain_mask,
                                        slice_mask,
                                        slice_mask.GetSize(),
                                        destinationIndex = [0, 0, slice_id])
        slice_id -= 1

    #sitk.WriteImage(outside_brain_mask, "outside_brain_mask_no_neck.mha")

    # HU value of air (1000) + threshold to account for noise (100) + value to stay above the max brain HU value.
    hu_delta = 1000 + 100 + (max_brain_HU + abs(min_brain_HU))
    aux_image = outside_brain_mask * hu_delta + input_image

    #sitk.WriteImage(aux_image, "before_RG.mha")

    # Get a seed inside the brain
    label_info_filter = sitk.LabelStatisticsImageFilter()
    label_info_filter.Execute(outside_brain_mask, outside_brain_mask)
    bound_box = label_info_filter.GetBoundingBox(0)
    seed_x = int((bound_box[1] - bound_box[0]) / 2 + bound_box[0])
    seed_y = int((bound_box[3] - bound_box[2]) / 2 + bound_box[2])
    seed_z = int((bound_box[5] - bound_box[4]) / 2 + bound_box[4])
    while outside_brain_mask[seed_x, seed_y, seed_z] != 0: # Put seed inside the mask if necessary.
        seed_z += 1
    seed = (seed_x, seed_y, seed_z)

    region_growing_filter = sitk.ConnectedThresholdImageFilter()
    region_growing_filter.SetLower(min_brain_HU)
    region_growing_filter.SetUpper(max_brain_HU)
    region_growing_filter.SetSeed(seed)
    aux_image = region_growing_filter.Execute(aux_image)
    aux_image = sitk.BinaryFillhole(aux_image)

    #sitk.WriteImage(aux_image, "D:/MRCLEAN/REGISTRY/CTA_THIN/R2061/after_RG.mha")

    return sitk.Cast(aux_image, sitk.sitkUInt8)
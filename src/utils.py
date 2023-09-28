import numpy as np
import torch
import random
import colorsys
import cv2


def calculate_iou(prediction, mask):
    intersection = prediction * mask
    union = prediction + mask - intersection
    return intersection.sum() / (union.sum() + 1e-7)


def get_crops_coords(image_size, patch_size, num_patchs_per_side):
    h, w = image_size
    if num_patchs_per_side == 1:
        x_step_size = y_step_size = 0
    else:
        x_step_size = (w - patch_size) // (num_patchs_per_side - 1)
        y_step_size = (h - patch_size) // (num_patchs_per_side - 1)
    crops_coords = []
    for i in range(num_patchs_per_side):
        for j in range(num_patchs_per_side):
            y_start, y_end, x_start, x_end = (
                i * y_step_size,
                i * y_step_size + patch_size,
                j * x_step_size,
                j * x_step_size + patch_size,
            )
            crops_coords.append([y_start, y_end, x_start, x_end])
    return crops_coords


def get_random_crop_coordinates(crop_scale_range, image_width, image_height):
    rand_number = random.random()
    rand_number *= crop_scale_range[1] - crop_scale_range[0]
    rand_number += crop_scale_range[0]
    patch_size = int(rand_number * min(image_width, image_height))
    if patch_size != min(image_width, image_height):
        x_start = random.randint(0, image_width - patch_size)
        y_start = random.randint(0, image_height - patch_size)
    else:
        x_start = 0
        y_start = 0
    return x_start, x_start + patch_size, y_start, y_start + patch_size


def generate_distinct_colors(n):
    colors = []
    if n == 1:
        return [(255, 255, 255)]
    for i in range(n):
        hue = i / n
        saturation = 0.9
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        scaled_rgb = tuple(int(x * 255) for x in rgb)
        colors.append(scaled_rgb)
    return colors


def get_boundry_and_eroded_mask(mask):
    kernel = np.ones((7, 7), np.uint8)
    eroded_mask = np.zeros_like(mask)
    boundry_mask = np.zeros_like(mask)
    for part_mask_idx in np.unique(mask)[1:]:
        part_mask = np.where(mask == part_mask_idx, 1, 0)
        part_mask_erosion = cv2.erode(part_mask.astype(np.uint8), kernel, iterations=1)
        part_boundry_mask = part_mask - part_mask_erosion
        eroded_mask = np.where(part_mask_erosion > 0, part_mask_idx, eroded_mask)
        boundry_mask = np.where(part_boundry_mask > 0, part_mask_idx, boundry_mask)
    return eroded_mask, boundry_mask


def get_colored_segmentation(mask, boundry_mask, image, colors):
    boundry_mask_rgb = 0
    if boundry_mask is not None:
        boundry_mask_rgb = torch.repeat_interleave(boundry_mask[None, ...], 3, 0).type(
            torch.float
        )
        for j in range(3):
            for i in range(1, len(colors) + 1):
                boundry_mask_rgb[j] = torch.where(
                    boundry_mask_rgb[j] == i,
                    colors[i - 1][j] / 255,
                    boundry_mask_rgb[j],
                )
    mask_rgb = torch.repeat_interleave(mask[None, ...], 3, 0).type(torch.float)
    for j in range(3):
        for i in range(1, len(colors) + 1):
            mask_rgb[j] = torch.where(
                mask_rgb[j] == i, colors[i - 1][j] / 255, mask_rgb[j]
            )
    if boundry_mask is not None:
        return (boundry_mask_rgb * 0.6 + mask_rgb * 0.3 + image * 0.4).permute(1, 2, 0)
    else:
        return (mask_rgb * 0.6 + image * 0.4).permute(1, 2, 0)

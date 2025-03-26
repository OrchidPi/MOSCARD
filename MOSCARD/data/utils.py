import numpy as np
import cv2


def border_pad(image, cfg):
    h, w, c = image.shape

    if cfg.border_pad == 'zero':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=0.0)
    elif cfg.border_pad == 'pixel_mean':
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode='constant',
                       constant_values=cfg.pixel_mean)
    else:
        image = np.pad(image, ((0, cfg.long_side - h),
                               (0, cfg.long_side - w), (0, 0)),
                       mode=cfg.border_pad)

    return image


def fix_ratio(image, cfg, apply_augmentations=True):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = cfg.long_side
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = cfg.long_side
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    if apply_augmentations:
        image = border_pad(image, cfg)

    return image


def transform(image, cfg, apply_augmentations=True):
    """
    Transform the image. Optionally apply augmentations like Gaussian Blur and Histogram Equalization.
    
    :param image: Input image (numpy array).
    :param cfg: Configuration object with settings for transformations.
    :param apply_augmentations: If False, don't apply augmentations (e.g., for target images).
    :return: Transformed image (numpy array).
    """
    assert image.ndim == 2, "image must be gray image"
    # Only apply augmentations if requested
    if apply_augmentations:
        if cfg.use_equalizeHist:
            image = cv2.equalizeHist(image)

        if cfg.gaussian_blur > 0:
            image = cv2.GaussianBlur(
                image,
                (cfg.gaussian_blur, cfg.gaussian_blur), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = fix_ratio(image, cfg, apply_augmentations=True)
    # augmentation for train or co_train

    # normalization
    image = image.astype(np.float32) - cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    if cfg.pixel_std:
        image /= cfg.pixel_std

    #image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    image = image.transpose((2, 0, 1))

    return image

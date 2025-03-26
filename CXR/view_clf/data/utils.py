import numpy as np
import cv2


def border_pad(image):
    h, w, c = image.shape

    image = np.pad(image, ((0, 512 - h),
                               (0, 512 - w), (0, 0)),
                       mode='constant',
                       constant_values=128.0)

    return image


def fix_ratio(image):
    h, w, c = image.shape

    if h >= w:
        ratio = h * 1.0 / w
        h_ = 512
        w_ = round(h_ / ratio)
    else:
        ratio = w * 1.0 / h
        w_ = 512
        h_ = round(w_ / ratio)

    image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
    image = border_pad(image)

    return image


def transform(image):
    assert image.ndim == 2, "image must be gray image"
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image,(3, 3), 0)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #image = fix_ratio(image, cfg)
    # augmentation for train or co_train

    # normalization
    #image = image.astype(np.float32) - cfg.pixel_mean
    # vgg and resnet do not use pixel_std, densenet and inception use.
    #if cfg.pixel_std:
    #    image /= cfg.pixel_std
    # normal image tensor :  H x W x C
    # torch image tensor :   C X H X W
    #image = image.transpose((2, 0, 1))

    return image

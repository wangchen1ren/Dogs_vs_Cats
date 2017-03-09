import numpy as np
from skimage import transform
from skimage.io import imread

MEAN_PIXEL = [127, 127, 127]


def crop_image(image, dshape):
    factor = float(min(dshape[:2])) / min(image.shape[:2])
    new_size = [int(image.shape[0] * factor), int(image.shape[1] * factor)]
    if new_size[0] < dshape[0]:
        new_size[0] = dshape[0]
    if new_size[1] < dshape[0]:
        new_size[1] = dshape[0]
    resized_image = transform.resize(image, new_size)
    sample = np.asarray(resized_image) * 256
    if dshape[0] < sample.shape[0] or dshape[1] < sample.shape[1]:
        xx = int((sample.shape[0] - dshape[0]))
        yy = int((sample.shape[1] - dshape[1]))
        xstart = xx / 2
        ystart = yy / 2
        xend = xstart + dshape[0]
        yend = ystart + dshape[1]
        sample = sample[xstart:xend, ystart:yend, :]
    return sample


def preprocess_image(path, dshape):
    image = imread(path)
    image = crop_image(image, dshape=dshape)
    image -= MEAN_PIXEL
    return image


def postprocess_image(image, dshape):
    image = np.reshape(image, dshape) + MEAN_PIXEL
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

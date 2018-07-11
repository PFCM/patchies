"""
helpers for preprocessing the cats dataset
"""

from itertools import islice

import cv2
import numpy as np


def _read_annotation(path):
    """read a ".cat" file containing pairs of points. We're not too worried
    about what they are."""
    with open(path) as infile:
        annotation = infile.read().strip().split(' ')

    # first should be the number of points, we'll just ignore it
    # the others are pairs
    nums = np.asarray([int(dim) for dim in annotation[1:]])
    return nums.reshape((-1, 2))


def load_cat_and_face_bbox(cat_path):
    """Load a cat image and the associated annotations, take the bounding box
    of the face annotations. The bounding box will cover the mouth and eyes
    only, so you will probably want to pad it out."""
    annotation_path = cat_path + '.cat'
    cat_im = cv2.imread(cat_path, 1)
    face_points = _read_annotation(annotation_path)

    bbox = np.asarray([[np.min(face_points[:, 0]),
                        np.min(face_points[:, 1])],
                       [np.max(face_points[:, 0]),
                        np.max(face_points[:, 1])]])

    bbox = np.clip(bbox, 0, cat_im.shape[:2])

    return cat_im, bbox


def square_bbox(bbox, side):
    """Reshape a bounding box so that it has the same center but all sides are
    size `side`."""
    bbox_sides = bbox[1, :] - bbox[0, :]
    pads = side - bbox_sides
    bbox[0, :] -= pads // 2
    bbox[1, :] = bbox[0, :] + side
    return bbox


def crop_to_square_bbox(img, bbox):
    """Crop a square out of an image in a way related to `bbox`. Takes the
    following steps:
    - make the smaller side of the bbox the same size as the largest
      (keeping the centre the same)
    - if this box extends beyond the image, try and recentre it until it fits
    - if the box is just bigger than the image try take the largest square that
      fits
    """
    max_side = np.max(bbox[1, :] - bbox[0, :])
    img_bbox = np.array([[0, 0], list(img.shape[:2])])
    min_img = np.min(img_bbox[1, :] - img_bbox[0, :])

    bbox = square_bbox(bbox, min(max_side, min_img))

    # potentially recentre if it's wandered off the edges
    if np.any(bbox < 0):
        bbox += np.abs(bbox[0, :]) * (bbox[0, :] < 0)
    if np.any(bbox[1, :] >= img_bbox[1, :]):
        bbox -= np.clip(bbox[1, :] - img_bbox[1, :], 0, None)

    return img[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0], :]


def pad_bbox(bbox, pad):
    """Pad a bounding box by a certain fraction in each direction. For example
    0.5 would make it half as big again"""
    pad_pixels = ((bbox[1, :] - bbox[0, :]) * pad).astype(np.int)
    bbox[0, :] -= pad_pixels
    bbox[1, :] += pad_pixels
    return bbox


def scan(func, initial):
    """generate repeated applications of `func`"""
    current = initial
    while True:
        yield current
        current = func(current)


def process_cat(cat_path, face_pad=0.25, final_shape=(64, 64)):
    """process a single cat image into a final shape. Involves the following
    steps:
    - load the image
    - load the annotation
    - compute the bounding box of the face annotations
    - pad the bounding box
    - crop the image
    - resize the cropped image
    """
    img, bbox = load_cat_and_face_bbox(cat_path)
    bbox = pad_bbox(bbox, face_pad)
    img = crop_to_square_bbox(img, bbox)
    img = cv2.resize(img, final_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # return all four 90 degree rotations
    return tuple(islice(scan(np.rot90, img), 4))

"""
Pipeline for actually processing images.
"""
import contextlib
import logging
import multiprocessing
import os
from functools import partial
from itertools import chain

import numpy as np
import observations
from PIL import Image

from patchies.cats import process_cat
from patchies.index import img_index


# stolen from scikit-image: http://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.view_as_blocks
def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).
    Blocks are non-overlapping views of the input array.
    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    block_shape : tuple
        The shape of the block. Each dimension must divide evenly into the
        corresponding dimensions of `arr_in`.
    Returns
    -------
    arr_out : ndarray
        Block view of the input array.  If `arr_in` is non-contiguous, a copy
        is made.
    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_blocks
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = view_as_blocks(A, block_shape=(2, 2))
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13
    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],
           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],
           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],
           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = view_as_blocks(A, block_shape=(1, 2, 2))
    >>> B.shape
    (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]  # doctest: +NORMALIZE_WHITESPACE
    array([[[[52, 53],
             [58, 59]]],
           [[[76, 77],
             [82, 83]]]])
    """
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    arr_in = np.ascontiguousarray(arr_in)

    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = np.lib.stride_tricks.as_strided(
        arr_in, shape=new_shape, strides=new_strides)

    return arr_out


def small32_imagenet(path):
    """get the data"""
    ims = observations.small32_imagenet(path)
    data = np.concatenate(ims, 0)
    return data.reshape((data.shape[0], -1))


def _process_single_celeba(fname):
    """load an image, process it"""
    img = Image.open(fname)
    img = img.resize((44, 54))
    img = np.array(img)
    return img[16:48, 6:38, :]


def _count_loader(gen):
    """print some progress"""
    for i, item in enumerate(gen):
        yield item
        print('\rprocessed {}'.format(i), end='', flush=True)
    print()


def cats(cats_path, patch_size, outpath):
    """Get the kaggle cats dataset. Needs login, so you'll have to
    have previously downloaded it."""
    datafile = os.path.join(outpath, 'cats-{}.npy'.format(patch_size))
    if not os.path.exists(datafile):
        # then we'll have to load in cats and make them the same size
        # the cat .jpg are stored in a few folders so we'll look recursively
        fnames = (os.path.join(dirpath, fname)
                  for dirpath, _, fnames in os.walk(cats_path)
                  for fname in fnames if fname.endswith('.jpg'))
        logging.info('no preprocessed cats, processing now')

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            processor = partial(
                process_cat, final_shape=(patch_size, patch_size))
            ims = np.stack(
                chain.from_iterable(
                    pool.imap(processor, _count_loader(fnames), 100)))
        np.save(datafile, ims)
    else:
        logging.info('found preprocessed cats')
        ims = np.load(datafile)

    return ims.reshape((ims.shape[0], -1))


def celeba(path):
    """get celeba, needs some preprocessing because it comes in
    178x218 jpegs. To get them to a more useful 32 by 32 we'll
    resize them down by a factor of 4 and then take a central
    crop."""
    # check it we've done the pre-processing already
    datafile = os.path.join(path, 'celeba.npy')
    if not os.path.exists(datafile):
        logging.info('loading and preprocessing celeba')
        # make sure it's downloaded
        _ = observations.celeba(path)
        datadir = os.path.join(path, 'img_align_celeba')
        filenames = (os.path.join(datadir, fname)
                     for fname in os.listdir(datadir)
                     if fname.endswith('.jpg'))
        filenames = _count_loader(filenames)
        with multiprocessing.Pool(8) as pool:
            ims = np.stack(pool.imap(_process_single_celeba, filenames, 50))
        np.save(datafile, ims)
    else:
        logging.info('found preprocessed celeba')
        ims = np.load(datafile)

    return ims.reshape((ims.shape[0], -1))


def cifar100(path):
    """get a cifar 100"""
    (train, _), (test, _) = observations.cifar100(path)
    data = np.concatenate((train, test), 0)
    data = data.transpose(0, 2, 3, 1)
    # data = data[:, ::2, ::2, :]
    return data.reshape((data.shape[0], -1))


def slice_params(axis, factor):
    """Get the start and stop indices to slice a dimension of size `axis` into
    a multiple of `factor`, keeping it centered."""
    new_size = (axis // factor) * factor
    start = (axis - new_size) // 2
    end = axis - (axis - new_size - start)
    return start, end


def make_mosaic(index, img, patch_size, data):
    """Replace all the patches in `img` with images pulled out of `index`."""
    img_patches = view_as_blocks(img, (patch_size, patch_size, 3))
    patches_x, patches_y = img_patches.shape[:2]
    img_patches = img_patches.reshape((patches_x * patches_y, -1))

    queries = img_patches.astype(np.float32) / 127.0 - 1
    neighbours = index.knnQueryBatch(queries, k=1)
    ids, dists = zip(*neighbours)
    ids = np.asarray(ids)
    logging.debug('ids shape: %s', np.asarray(ids).shape)

    if ids.shape[-1] == 0:
        ids = np.zeros((ids.shape[0], 1), dtype=np.int32)
    ids = np.squeeze(ids, 1)

    new_patches = data[ids]
    # getting them back into the right layout is surprisingly tricky
    new_patches = new_patches.reshape((patches_x, patches_y, patch_size,
                                       patch_size, 3))
    new_patches = new_patches.swapaxes(1, 2)
    new_patches = new_patches.reshape(patches_x * patch_size,
                                      patches_y * patch_size, 3)

    logging.debug('new shape %s', new_patches.shape)
    logging.debug('min: %d, max: %d, mean: %d', new_patches.min(),
                  new_patches.max(), new_patches.mean())

    return new_patches

"""
Pipeline for actually processing images.
"""
import contextlib
import logging
import multiprocessing
import os
from functools import partial
from itertools import chain

import click
import cv2
import numpy as np
import observations
from skimage.util import view_as_blocks

from patchies.cats import process_cat
from patchies.index import img_index


def small32_imagenet(path):
    """get the data"""
    ims = observations.small32_imagenet(path)
    data = np.concatenate(ims, 0)
    return data.reshape((data.shape[0], -1))


def _process_single_celeba(fname):
    """load an image, process it"""
    img = cv2.imread(fname, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (44, 54))
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


def _slice_params(axis, factor):
    """Get the start and stop indices to slice a dimension of size `axis` into
    a multiple of `factor`, keeping it centered."""
    new_size = (axis // factor) * factor
    start = (axis - new_size) // 2
    end = axis - (axis - new_size - start)
    return start, end


def get_frames(device, size_factor):
    """Yield video frames from a cv video capture. Slices both dimensions to be
    a multiple of `size_factor`"""
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        logging.error('could not open video device %s', device)
        rval = False
    else:
        rval, frame = cap.read()

    while rval:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # make it smaller for now
        frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
        x_start, x_end = _slice_params(frame.shape[0], size_factor)
        y_start, y_end = _slice_params(frame.shape[1], size_factor)
        frame = frame[x_start:x_end, y_start:y_end, :]
        yield frame
        rval, frame = cap.read()

    cap.release()


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


@contextlib.contextmanager
def index_from_ctx(ctx):
    """call the index context manager with args from context"""
    with img_index(
            ctx.obj['imdir'],
            ctx.obj['loader'],
            construction_args=ctx.obj['creation_params'],
            query_args=ctx.obj['query_params']) as stuff:
        yield stuff


@click.group()
@click.pass_context
@click.option('--imdir', default=None, help='directory to store images')
@click.option('--cores', default=8, help='number of threads to use', type=int)
@click.option(
    '--dataset',
    default='cifar100',
    help='dataset to use',
    type=click.Choice(['cifar100', 'imagenet', 'celeba', 'cats']))
@click.option('--cats_path', help='path to downloaded cats data')
@click.option(
    '--patch_size', help='size of patches to replace', default=32, type=int)
def cli(ctx, imdir, cores, dataset, cats_path, patch_size):
    """Run on some things."""
    logging.basicConfig(level=logging.INFO)

    if imdir is None:
        imdir = os.path.join(
            os.path.dirname(__file__), 'img', '{}-{}-index.bin'.format(
                dataset, patch_size))

    if imdir is None:
        imdir = os.path.join(
            os.path.dirname(__file__), 'img',
            dataset + '-{}-index.bin'.format(patch_size))

    creation_params = {
        'M': 50,
        'indexThreadQty': cores,
        'efConstruction': 400,
        'post': 2,
        'skip_optimized_index':
        1  # can't get the data out of python bindings anyway...
    }
    query_params = {'efSearch': 100}

    if dataset == 'cifar100':
        loader = cifar100
    elif dataset == 'imagenet':
        loader = small32_imagenet
    elif dataset == 'cats':
        loader = partial(cats, cats_path, patch_size)
    else:
        loader = celeba

    if patch_size != 32 and dataset != 'cats':
        raise ValueError(
            'only cats dataset supports patch sizes that are not 32')

    # and set up the args in the context for the subcommands
    ctx.obj = {}
    ctx.obj['imdir'] = imdir
    ctx.obj['cores'] = cores
    ctx.obj['dataset'] = dataset
    ctx.obj['cats_path'] = cats_path
    ctx.obj['patch_size'] = patch_size
    ctx.obj['loader'] = loader
    ctx.obj['creation_params'] = creation_params
    ctx.obj['query_params'] = query_params


@cli.command()
@click.pass_context
@click.option('--outfile', default='out.png', help='output path to write')
@click.argument('filename', type=click.Path(exists=True))
def offline(ctx, outfile, filename):
    """Run the thing on a single image"""
    logging.basicConfig(level=logging.INFO)
    input_img = cv2.imread(filename)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    logging.info('input image: %dx%d', input_img.shape[0], input_img.shape[1])

    x_start, x_end = _slice_params(input_img.shape[0], ctx.obj['patch_size'])
    y_start, y_end = _slice_params(input_img.shape[1], ctx.obj['patch_size'])
    input_img = input_img[x_start:x_end, y_start:y_end, :]
    logging.info('sliced input: %dx%d', input_img.shape[0], input_img.shape[1])

    with index_from_ctx(ctx) as stuff:
        index, data = stuff
        new_img = make_mosaic(index, input_img, ctx.obj['patch_size'], data)
    logging.info('output image: %dx%d', new_img.shape[0], new_img.shape[1])
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outfile, new_img)


@cli.command()
@click.pass_context
@click.option('--device', help='video device to use', default='0')
def online(ctx, device):
    """Run the thing on a video."""
    logging.basicConfig(level=logging.INFO)
    patch_size = ctx.obj['patch_size']

    with index_from_ctx(ctx) as stuff:
        index, data = stuff
        cv2.namedWindow('patchies')
        for frame in get_frames(device, patch_size):
            img = make_mosaic(index, frame, patch_size, data)

            big_im = np.concatenate(
                (img, np.zeros((img.shape[0], 1, 3), dtype=np.uint8), frame),
                axis=1)

            cv2.imshow('patchies', cv2.cvtColor(big_im, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

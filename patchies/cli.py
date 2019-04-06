"""
Command line interface for image processing.
"""

import contextlib
import logging
import os
from functools import partial

import click
import cv2
import numpy as np

from patchies.index import img_index
from patchies.pipeline import (cats, celeba, cifar100, make_mosaic,
                               slice_params, small32_imagenet)


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
        x_start, x_end = slice_params(frame.shape[0], size_factor)
        y_start, y_end = slice_params(frame.shape[1], size_factor)
        frame = frame[x_start:x_end, y_start:y_end, :]
        yield frame
        rval, frame = cap.read()

    cap.release()


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

    x_start, x_end = slice_params(input_img.shape[0], ctx.obj['patch_size'])
    y_start, y_end = slice_params(input_img.shape[1], ctx.obj['patch_size'])
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

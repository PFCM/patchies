"""
Structures for loading and saving an index over the images.
"""
import contextlib
import logging
import os
import time

import numpy as np

import nmslib


@contextlib.contextmanager
def img_index(index_path,
              dataset_getter,
              method='hnsw',
              space='l2',
              data_type=nmslib.DataType.DENSE_VECTOR,
              construction_args=None,
              query_args=None):
    """
    Loads an index over something. If `index_path` exists we just load that,
    otherwise we call `dataset_getter` with a path which should (down)load the
    images and we will build the index. This could take a while.
    """
    index = nmslib.init(
        method=method,
        space=space,
        data_type=data_type,
        dtype=nmslib.DistType.FLOAT)
    data = dataset_getter(os.path.dirname(index_path))

    if os.path.exists(index_path):
        logging.info('found existing index %s', index_path)
        input_data = data.astype(np.float32) / 127.0 - 1
        index.addDataPointBatch(input_data)
        index.loadIndex(index_path)
    else:
        logging.info('no existing index, getting data')
        print(data.shape, data.dtype, np.min(data), np.max(data))
        logging.info('got data, building index')
        input_data = data.astype(np.float32) / 127.0 - 1
        index.addDataPointBatch(input_data)

        start = time.time()
        index.createIndex(construction_args, print_progress=True)
        end = time.time()
        logging.info('built index in %fs', end - start)
        logging.info('saving index')
        index.saveIndex(index_path)

    index.setQueryTimeParams(query_args)

    yield index, data

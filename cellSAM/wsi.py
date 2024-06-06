import functools
import logging
import operator

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
import numpy as np

from dask_image.ndmeasure._utils import _label
from sklearn import metrics as sk_metrics

from .model import segment_cellular_image

# Initialize a Dask cluster and specify GPU IDs for each worker
def setup_cluster(gpu_ids):
    cluster = LocalCluster(n_workers=len(gpu_ids), threads_per_worker=1,
                           resources={f'GPU{i}': 1 for i in gpu_ids})
    client = Client(cluster)
    return client, {i: gpu_id for i, gpu_id in enumerate(gpu_ids)}

def segment_chunk(chunk, gpu_id, model=None, **kwargs):
    """ Segments an individual chunk using a specific GPU.
    Args:
        chunk (np.array): Image data to segment.
        gpu_id (int): Identifier for the GPU to use.
    """
    import cupy as cp  # Using CuPy for GPU array management
    with cp.cuda.Device(gpu_id):
        mask = segment_cellular_image(chunk, model, device=f'cuda:', **kwargs)[0]
    return mask.astype(np.int64), mask.max()

def segment_wsi(image, overlap, iou_depth, iou_threshold, gpu_ids, **segmentation_kwargs):
    client, worker_to_gpu = setup_cluster(gpu_ids)  # Setup a Dask cluster with specified GPUs

    if image.ndim == 2:
        image = image[..., None]

    image = da.asarray(image)
    image = image.rechunk({-1: -1})  # Keep color channel together

    depth = (overlap, overlap)
    boundary = "reflect"
    image = da.overlap.overlap(image, depth + (0,), boundary)

    block_iter = zip(
        np.ndindex(*image.numblocks),
        map(
            functools.partial(operator.getitem, image),
            da.core.slices_from_chunks(image.chunks),
        ),
    )

    labeled_blocks = np.empty(image.numblocks[:-1], dtype=object)
    total = None

    for index, input_block in block_iter:
        worker_index = index[0] % len(gpu_ids)  # Distribute blocks across specified GPUs
        gpu_id = worker_to_gpu[worker_index]
        labeled_block, n = dask.delayed(segment_chunk, nout=2)(
            input_block,
            gpu_id,
            model=None,
            **segmentation_kwargs
        )

        shape = input_block.shape[:-1]
        labeled_block = da.from_delayed(labeled_block, shape=shape, dtype=np.int32)
        n = dask.delayed(np.int32)(n)
        n = da.from_delayed(n, shape=(), dtype=np.int32)

        total = n if total is None else total + n

        block_label_offset = da.where(labeled_block > 0, total, np.int32(0))
        labeled_block += block_label_offset

        labeled_blocks[index[:-1]] = labeled_block
        total += n

    block_labeled = da.block(labeled_blocks.tolist())

    # Rest of your function continues without changes
    # Note to include the cleanup of Dask client and cluster:
    client.close()
    client.cluster.close()


    depth = da.overlap.coerce_depth(len(depth), depth)

    if np.prod(block_labeled.numblocks) > 1:
        iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

        if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
            raise ValueError("iou_depth (%s) > depth (%s)" % (iou_depth, depth))

        trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
        block_labeled = da.overlap.trim_internal(
            block_labeled, trim_depth, boundary=boundary
        )
        block_labeled = link_labels(
            block_labeled,
            total,
            iou_depth,
            iou_threshold=iou_threshold,
        )

        block_labeled = da.overlap.trim_internal(
            block_labeled, iou_depth, boundary=boundary
        )

    else:
        block_labeled = da.overlap.trim_internal(
            block_labeled, depth, boundary=boundary
        )

    return block_labeled



def link_labels(block_labeled, total, depth, iou_threshold=1):
    """
    Build a label connectivity graph that groups labels across blocks,
    use this graph to find connected components, and then relabel each
    block according to those.
    """
    label_groups = label_adjacency_graph(block_labeled, total, depth, iou_threshold)
    new_labeling = _label.connected_components_delayed(label_groups)
    return _label.relabel_blocks(block_labeled, new_labeling)


def label_adjacency_graph(labels, nlabels, depth, iou_threshold):
    all_mappings = [da.empty((2, 0), dtype=np.int32, chunks=1)]

    slices_and_axes = get_slices_and_axes(labels.chunks, labels.shape, depth)
    for face_slice, axis in slices_and_axes:
        face = labels[face_slice]
        mapped = _across_block_iou_delayed(face, axis, iou_threshold)
        all_mappings.append(mapped)

    i, j = da.concatenate(all_mappings, axis=1)
    result = _label._to_csr_matrix(i, j, nlabels + 1)
    return result


def _across_block_iou_delayed(face, axis, iou_threshold):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_iou)
    grouped = _across_block_label_grouping_(face, axis, iou_threshold)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=np.int32)


def _across_block_label_iou(face, axis, iou_threshold):
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)

    intersection = sk_metrics.confusion_matrix(face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)

    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection

    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0)

    labels0, labels1 = np.nonzero(iou >= iou_threshold)

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])

    valid = np.all(grouped != 0, axis=0)  # Discard any mappings with bg pixels
    return grouped[:, valid]


def get_slices_and_axes(chunks, shape, depth):
    ndim = len(shape)
    depth = da.overlap.coerce_depth(ndim, depth)
    slices = da.core.slices_from_chunks(chunks)
    slices_and_axes = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - 2 * depth[ax], sl[ax].stop + 2 * depth[ax]
            )
            slices_and_axes.append((tuple(slice_to_append), ax))
    return slices_and_axes
"""
Generic tools for distributing computationally intensive tasks across multiple threads.
"""

import os
import numpy as np
import shutil
from tempfile import mkdtemp
import multiprocessing as mp
from tqdm import tqdm

from hylite import HyCloud, HyImage
from hylite import io

def _split(data, nchunks):
    """
    Split the specified HyCloud instance into a number of chunks.

    *Arguments*:
     - data = the complete HyData object to copy and split.
     - nchunks = the number of chunks to split into.
    *Returns*:
     - a list of split
    """

    if isinstance(data, HyCloud):  # special case for hyperclouds - split xyz, rgb and normals too
        chunksize = int(np.floor(data.point_count() / nchunks))
        chunks = [(i * chunksize, (i + 1) * chunksize) for i in range(nchunks)]
        chunks[-1] = (chunks[-1][0], data.point_count())  # expand last chunk to include remainder

        # split points
        xyz = [data.xyz[c[0]:c[1], :].copy() for c in chunks]

        # split data
        bands = [None for c in chunks]
        if data.has_bands():
            X = data.get_raveled().copy()
            bands = [X[c[0]:c[1], :] for c in chunks]

        # split rgb
        rgb = [None for c in chunks]
        if data.has_rgb():
            rgb = [data.rgb[c[0]:c[1], :].copy() for c in chunks]

        # split normals
        normals = [None for c in chunks]
        if data.has_normals():
            normals = [data.normals[c[0]:c[1], :].copy() for c in chunks]

        return [HyCloud(xyz[i],
                        rgb=rgb[i],
                        normals=normals[i],
                        bands=bands[i],
                        header=data.header.copy()) for i in range(len(chunks))]

    else:  # just split data (for HyImage and other types)
        X = data.get_raveled().copy()
        chunksize = int(np.floor(X.shape[0] / nchunks))
        chunks = [(i * chunksize, (i + 1) * chunksize) for i in range(nchunks)]
        chunks[-1] = (chunks[-1][0], X.shape[0])  # expand last chunk to include remainder

        out = []
        for c in chunks:
            _o = data.copy(data=False)  # create copy
            _o.data = X[c[0]:c[1], :][:,None,:]
            out.append(_o)

        return out

def _merge(chunks, shape):
    """
    Merge a list of HyData objects into a combined one (aka. do the opposite of split(...)).

    *Arguments*:
     - chunks = a list of HyData chunks to merge.
     - shape = the output data shape.
    *Returns*: a single merged HyData instance (of the same type as the input).
               The header of this instance will be a copy of chunks[0].header.
    """

    # merge data
    X = np.vstack([c.data for c in chunks])
    X = X.reshape((*shape, -1))

    if not isinstance(chunks[0], HyCloud): # easy!
        # make copy
        out = chunks[0].copy(data=False)
        out.data = X
        out.header = chunks[0].header.copy()
        return out
    else: # less easy
        xyz = np.vstack([c.xyz for c in chunks])
        rgb = None
        if chunks[0].has_rgb():
            rgb = np.vstack([c.rgb for c in chunks])
        normals = None
        if chunks[0].has_normals():
            normals = np.vstack([c.normals for c in chunks])
        return HyCloud( xyz, rgb=rgb, normals=normals, bands=X, header=chunks[0].header.copy())

def _call(func, path, arg, kwd, n):
    """
    This function will be called by each thread. It loads each data chunk from disk, runs the operation, then saves
    the results.
    """

    # print("Spawning thread %d." % n)
    # func, path, arg, kwd = args

    # load data chunk
    if '.ply' in path:
        data = io.loadCloudPLY(path)  # load point cloud
        result = func(data, *arg, **kwd)  # compute results
        assert isinstance(result, HyCloud), "Error - function %s does not return a HyCloud." % func
        io.saveCloudPLY(path, result)  # save point cloud
    else:
        data = io.load(path)  # load image
        result = func(data, *arg, **kwd)  # compute results
        assert isinstance(result, HyImage), "Error - function %s does not return a HyImage." % func
        io.save(path, result)  # save result

    return True  # done

def parallel_chunks(function, data, *args, **kwds):

    """
    Run a function that operates per-point or per-pixel on smaller chunks of a point cloud or image dataset
    in parallel. Only use for expensive operations as otherwise overheads (writing files to cache, spawning threads,
    loading files from cache) are too costly.

    *Arguments*:
     - function = the function to run on each chunk of the dataset. Must take a HyCloud or HyImage dataset as it's first
                  argument and also return a HyCloud or HyImage dataset (cf., mwl(...), get_hull_corrected(...)).
     - data = the HyCloud or HyImage instance to run the function on.
     - args = tuple of arguments to pass to the function.

     **Keywords**:
      - nthreads = the number of threads to spawn. Default is the number of cores - 2. Negative numbers will be subtracted
                   from the number of cores.
      - any other keywords are passed to the function
    """
    assert isinstance(data, HyCloud) or isinstance(data, HyImage)

    # get number of threads
    if 'nthreads' in kwds:
        nthreads = kwds['nthreads']
        del kwds['nthreads']
    else:
        nthreads = -2
    if nthreads < 1:
        nthreads = os.cpu_count() - nthreads
    assert nthreads > 0, "Error - cannot spawn %d threads" % nthreads

    assert isinstance(nthreads, int), "Error - nthreads must be an integer."
    assert nthreads is not None, "Error - could not identify CPU count. Please specify nthreads keyword."

    # split data into chunks
    shape = data.data.shape[:-1]  # store shape (important for images)
    chunks = _split(data, nthreads)

    # dump chunks into temp directory
    pth = mkdtemp()  # make temp directory
    print("Writing thread cache to %s:" % pth)

    # dump clouds to directory
    paths = []
    for i, c in enumerate(chunks):

        if isinstance(c, HyCloud):
            p = os.path.join(pth, '%d.ply' % i)
            io.saveCloudPLY(p, c)
        else:
            p = os.path.join(pth, '%d.hdr' % i)
            io.save(p, c)
        paths.append(p)

    # make sure we don't multithread twice when using advanced scipy/numpy functions...
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'

    # spawn worker processes
    P = [mp.Process(target=_call, args=(function, p, args, kwds, i)) for i, p in enumerate(paths)]
    try:
        for p in P:
            p.start()
        for p in P:
            p.join()

        # successs! load data again...
        if isinstance(data, HyCloud):
            chunks = [io.loadCloudPLY(p) for p in paths]
        else:
            chunks = [io.load(p) for p in paths]

        # remove temp directory
        shutil.rmtree(pth)  # delete temp directory
        print("Process complete (thread cache cleaned successfully).")
    except (KeyboardInterrupt, SystemExit) as e:
        print("Job cancelled. Cleaning temp directory... ", end='')
        shutil.rmtree(pth) # delete temp directory
        print("Done.")
        assert False, "Multiprocessing job cancelled by KeyboardInterrupt or SystemExit."
    except Exception as e:
        print("Error thrown. Cleaning temp directory... ", end='')
        shutil.rmtree(pth)  # delete temp directory
        print("Done.")
        raise e

    # re-enable scipy/numpy multithreading
    del os.environ['MKL_NUM_THREADS']
    del os.environ['OMP_NUM_THREADS']
    del os.environ['MKL_DYNAMIC']

    # merge back into one dataset
    out = _merge(chunks, shape=shape)
    return out

def _call2(func, in_paths, out_paths, kwd, n):

    for i, o in zip(in_paths, out_paths):  # loop through paths managed by this thread
        func(i, o, **kwd)  # call function

def parallel_datasets(function, in_paths, out_paths=None, nthreads=-2, **kwds):
    """
    Parallelise a single function across many HyData datasets.

    *Arguments*:
      - function = the function to run on each dataset. This should take an input path (string) as its first input
      and output path (also string) as its second output. Anything returned by the function will be ignored.
     - in_paths = a list of input paths, each of which will be passed to function in each thread.
     - out_paths = a list of corresponding output paths that each function should write to. Defaults to in_paths.
     - nthreads = the number of threads to spawn. Default is the number of cores - 2. Negative numbers are subtracted
                  from the total number of cores.
    *Keywords*:
     - any keywords are passed directly to function in each thread.

    *Returns*: Nothing.
    """

    assert isinstance(in_paths, list), "Error - in_paths must be a list of file paths (string)."

    if out_paths is None:
        out_paths = in_paths
    assert isinstance(out_paths, list), "Error - out_paths must be a list of file paths (string)."
    assert len(out_paths) == len(in_paths), "Error - length of input and output paths must match."

    # get number of threads
    assert isinstance(nthreads, int), "Error - nthreads must be an integer."
    if nthreads < 1:
        nthreads = os.cpu_count() - nthreads
    assert nthreads > 0, "Error - cannot spawn %d threads" % nthreads

    # distribute input paths across threads
    nthreads = min( len(in_paths), nthreads ) # avoid case where we have more threads than paths
    stride = int( len(in_paths) / nthreads )
    inP = []
    outP = []
    for i in range(nthreads):
        idx0 = i*stride
        idx1 = min( (i+1)*stride, len(in_paths) )
        inP.append( in_paths[idx0:idx1] )
        outP.append( out_paths[idx0:idx1] )
    for i in range(len(in_paths) % nthreads): # and add remainder
        inP[i].append(in_paths[-i-1])
        outP[i].append(out_paths[-i - 1])

    # make sure we don't multithread twice when using advanced scipy/numpy functions...
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'

    # spawn worker processes and wait for jobs to finish
    P = [mp.Process(target=_call2, args=(function, inP[i], outP[i], kwds, i)) for i in range(nthreads)]
    for p in P:
        p.start()
    for p in P:
        p.join()

    # re-enable scipy/numpy multithreading
    del os.environ['MKL_NUM_THREADS']
    del os.environ['OMP_NUM_THREADS']
    del os.environ['MKL_DYNAMIC']

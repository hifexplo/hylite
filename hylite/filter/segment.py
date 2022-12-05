"""
Functions for segmenting regions of images for individual analyses. E.g., segmenting core trays or hand samples
from images taken with a core scannner.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import hylite
import cv2
from matplotlib import path
from hylite import io
from tqdm import tqdm


def readAnnotations(annot, hsi, mask_patches=False, mask_panels=False, mask_points=False, mode='average', inplace=False):
    """
    Extract pixel positions and region spectra (e.g. calibration panels) based on an annotation image.
    This image should have three RGB bands (0 - 255) and will be parsed as follows:

     - [0,0,0] = black = retain data.
     - [255,255,255] = white = mask data (replace with nan).
     - [aaa,aaa,aaa] = grays = extract regions and add averaged spectra to header file indexed by grey value.
     - [0,255,255],[255,0,255],[255,255,0]  = cyan, magenta, yellow = extract calibration panels A, B and C (respectively).
     - [n,0,0] (reds), [0,n,0] (blues), [0,0,n] (greens) = extract pixel locations and add to header file as points A, B and C (respectively). These will be ordered based on the corresponding channel value.

    Args:
        annot: the annotations image (three channels, 0-255, uint8) matching the format described above.
        hsi: a hyperspectral image with matching x- and y- dimensions to the annotation image.
        mask_patches: True if patches should be removed from output. Default is False. [ i.e. these will be retained in the image ].
        mask_panels: True if panels should be removed from output. Default is False. [ i.e. these will be retained in the image ].
        mask_points: True if points should be removed from output. Default is False. [ i.e. these will be retained in the image ].
        mode: function used to aggregate pixel values. Options are 'average' (default) or 'median'.
        inplace: True if the hsi image should be modified in-place or copied (default).

    Returns:
        a copy of the input file (if inplace is False) with the extracted points and spectra added to the header file.
    """

    # check inputs
    assert isinstance(annot, hylite.HyImage), 'Error - annot (%s) is not a HyImage instance' % type(annot)
    assert isinstance(hsi, hylite.HyImage), 'Error - hsi (%s) is not a HyImage instance' % type(hsi)
    assert annot.band_count() == 3, 'error - annot has %d bands (should = 3).' % annot.band_count()
    assert annot.xdim() == hsi.xdim(), 'error - xdim %d != %d' % (annot.xdim(), hsi.xdim())
    assert annot.ydim() == hsi.ydim(), 'error - xdim %d != %d' % (annot.ydim(), hsi.ydim())

    if 'average' in mode.lower():
        agg = np.nanmean
    elif 'median' in mode.lower():
        agg = np.nanmedian
    else:
        assert False, "Error - %s is an unknown mode. Should be 'average' or 'median'." % mode

    # create output
    if inplace:
        out = hsi
    else:
        out = hsi.copy(data=True)

    # build masks
    diff = np.diff(annot.data, axis=-1)
    retain_mask = (annot.data == 0).all(axis=-1)
    remove_mask = (annot.data == 255).all(axis=-1)
    patch_mask = (diff == 0).all(axis=-1) & np.logical_not(remove_mask) & np.logical_not(retain_mask)
    panel_mask = np.sum(annot.data != 0, axis=-1) == 2
    point_mask = np.sum(annot.data != 0, axis=-1) == 1

    # extract patches
    patches = np.unique(annot.data[patch_mask, 0])
    for i, v in enumerate(patches):
        mask = patch_mask & (annot.data[..., 0] == v)
        if mask.any():
            S = agg(hsi.data[mask, :], axis=0)
            out.header['patch %d size' % int(v)] = np.sum(mask)
            out.header['patch %d spectra' % int(v)] = S

    # extract panels
    for i, v in zip(range(3), 'abc'):
        mask = panel_mask & (annot.data[..., i] == 0)
        if mask.any():
            S = agg(hsi.data[mask, :], axis=0)
            out.header['panel %s size' % v] = np.sum(mask)
            out.header['panel %s spectra' % v] = S

    # extract points
    for i, v in zip(range(3), 'abc'):
        mask = point_mask & (annot.data[..., i] != 0)
        if mask.any():
            vals = annot.data[mask, i]
            coords = np.argwhere(mask)
            idx = np.argsort(vals)
            out.header['points %s x' % v] = coords[idx, 0]
            out.header['points %s y' % v] = coords[idx, 1]

    # apply mask
    out.data[remove_mask, :] = np.nan
    if mask_patches:
        out.data[patch_mask, : ] = np.nan
    if mask_panels:
        out.data[panel_mask, : ] = np.nan
    if mask_points:
        out.data[point_mask, : ] = np.nan

    # return
    return out

# create foreground and background masks for grab cut algorithm
def label_blocks(image, fg=None, s=8, epad=20, boost=3, erode=3, bands=hylite.RGB, vb=True, **kwds):
    """
    Segments an image into background and different hand samples based on point labels.

    Args:
        image: the image to segment.
        fg: the foreground definition (labelled with sample IDs). If None (default) then samples are pulled from the header file.
        s: the number of pixels to label at each label point (if fg is not specified).
        epad: padding around the edge of the image to be specified as background. Default is 20 pixels.
        boost: multiplication factor to exaggurate foreground background contrast.
        erode: Amount of erosion to apply to remove small regions labelled as sample. Default is 3. Set to 0 to disable.
        bands: the bands to use for segmentation. Default is hylite.RGB.
        vb: True if figures should be generated for QAQC. Default is True.
        **kwds: keywords are passed to HyImage.quick_plot( ... ) if vb is true.
    """

    # create fg array?
    if fg is None:
        fg = np.zeros((image.data.shape[0], image.data.shape[1]), dtype=np.uint64)
        points = [ image.header.get_sample_points(s) for s in image.header.get_class_names() ]
        assert len(points) > 0, "Error - image has no defined sample points. Call image.pickSamples(...) before segmenting."

        for i, fp in enumerate(points):  # enumerate samples
            for p in fp:  # enumerate points
                fg[max(p[0] - s, 0): min(p[0] + s, fg.shape[0]),
                max(p[1] - s, 0): min(p[1] + s, fg.shape[1])] = i + 1

    mask = np.full((image.data.shape[0], image.data.shape[1]), cv2.GC_PR_BGD)
    mask[0:epad, :] = cv2.GC_BGD
    mask[-epad:-1] = cv2.GC_BGD
    mask[:, 0:epad] = cv2.GC_BGD
    mask[:, -epad:-1] = cv2.GC_BGD

    # add foreground pixels
    mask[fg != 0] = cv2.GC_FGD

    # extract RGB image and boost contrast
    rgb = image.data[..., [image.get_band_index(b) for b in bands]].copy()
    rgb -= np.nanmin(rgb)
    rgb /= np.nanmax(rgb)
    rgb *= boost
    rgb[rgb > 1] = 1
    rgb = (rgb * 255).astype(np.uint8)

    # do grab cut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(rgb, mask.astype(np.uint8), None, bgdModel, fgdModel, 5,
                                           cv2.GC_INIT_WITH_MASK)
    mask = (mask == cv2.GC_PR_FGD) + (mask == cv2.GC_FGD)  # convert to boolean foreground maks

    # erode
    if erode > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        mask = mask == 1

    # extract contours and create label mask
    contours,_ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    labels = np.zeros(mask.shape, dtype=np.int)
    xx, yy = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    xx = xx.flatten()
    yy = yy.flatten()
    points = np.vstack([xx, yy]).T  # coordinates of each pixel
    for cnt in contours:

        # convert to matplotlib path
        verts = np.array([cnt[:, 0, 1], cnt[:, 0, 0]]).T
        verts = np.vstack([verts, verts[0][None, :]])
        if verts.shape[0] < 3:
            continue

        pth = path.Path(verts, closed=True)

        # get internal labels
        inside = pth.contains_points(points)
        inside = inside.reshape((mask.shape[1], mask.shape[0])).T
        if inside.any():
            labels[inside] = np.max(fg[inside])  # fill label

    # plot
    if vb:
        fig, ax = image.quick_plot(bands, **kwds)
        ax.imshow(np.ma.MaskedArray(fg, fg == 0).T, alpha=0.8)
        ax.imshow(np.ma.MaskedArray(labels, labels == 0).T, alpha=0.5)
        fig.show()

    # create classification file
    cls = hylite.HyImage(labels[:, :, None])
    cls.header["file type"] = "ENVI Classification"

    n = int(np.max(fg))
    cls.header['classes'] = n
    names = image.header.get_class_names()
    if names is not None and len(names) == 0:
        names = [str(i) for i in range(1, n + 1)]
        cls.header['class names'] = ['background'] + names
    cmap = mpl.cm.get_cmap('viridis')
    cls.header['class lookup'] = (np.array([cmap(i)[:3] for i in np.linspace(0, 1, n)]).ravel() * 255).astype(np.uint8)

    return cls


# do extraction
def extract_tiles(image, labels, connected=False, ids=None, erode=0):
    """
    Extract tiles that each contain a single contiguous block based on the specified classification. Useful for
    extracting individual samples from core-scanner imagery.

    Args:
        image: the hyperspectral image to extract blocks from
        labels: a HyImage instance or numpy array with different samples labelled as non-0 values.
        connected: True if separate groups of pixels with the same label should be split
                   into different tiles using cv2.connectedComponents. Default is True.
        ids: a list of label ids to extract. If None (default), all non-zero classes will
             be extracted.
        erode: number of pixels to erode from mask before extracting tiles. Default is 0.
    """

    # get labels array
    if isinstance(labels, hylite.HyImage):
        labels = labels.data[..., 0]

    # get ids
    if ids is None:
        ids = np.unique(labels)
        ids = ids[ids != 0]  # remove zero

    # extract tiles
    tiles = []
    sampleID = []
    for i in ids:  # loop through labels

        # get mask
        mask = labels == i
        if not mask.any():
            continue  # can be the case if erode removed entire labels

        if erode > 0:
            mask = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode,))) == 1

        if not connected:
            # easy - don't bother to separate
            pixels = np.argwhere(mask)  # get valid pixels
            local = pixels - np.min(pixels, axis=0)[None, :]  # translate to get pixels in tile

            tile = hylite.HyImage(np.full((np.max(local[:, 0] + 1),
                                           np.max(local[:, 1] + 1), image.band_count()), np.nan))
            tile[local[:, 0], local[:, 1], :] = image.data[pixels[:, 0], pixels[:, 1], :]
            tile.set_wavelengths(image.get_wavelengths())

            # store label and tile
            sampleID.append(i)
            tiles.append(tile)
        else:
            # check connected components first
            nc, cl = cv2.connectedComponents(mask.astype(np.uint8))
            for n in range(nc):  # loop through components
                comp_mask = cl == n
                if not mask[comp_mask].all():
                    continue  # this component is background

                mask2 = mask & comp_mask

                # extact pixels
                pixels = np.argwhere(mask2)  # get valid pixels
                local = pixels - np.min(pixels, axis=0)[None, :]  # translate to get pixels in tile

                tile = hylite.HyImage(np.full((np.max(local[:, 0] + 1),
                                               np.max(local[:, 1] + 1), image.band_count()), np.nan))
                tile[local[:, 0], local[:, 1], :] = image.data[pixels[:, 0], pixels[:, 1], :]
                tile.set_wavelengths(image.get_wavelengths())

                # store label and tile
                sampleID.append(i)
                tiles.append(tile)

    return tiles, sampleID

def group_tiles( tiles, groups, ids=None, rotate=True, ignore=[] ):
    """
    Group a list of image tiles based on corresponding group IDs.

    Args:
        tiles: a list of HyImage tiles.
        groups: a list of integers corresponding to the grouping to arrange tiles by.
        ids: a list of sample IDs to create an id classification (or None).
        rotate: True if tiles should be rotated such that they're longer dimension aligns with the x-axis.
        ignore: a list of group id's that should be ignored/excluded from this grouping.
    Returns:
        A tuple containing:

         - tiled_image = a HyImage instance containing the arranged tiles.
         - tiled_label = a HyImage classification containing the class indices used to arrange the tiles.
         - [tiled_id] = a HyImage classification containing individual sample IDs (if ids is not None).
         - bounds = the bounds (x,y,width,height) of each group of tiles. Useful for labelling or separation.
     """

    # rotate tiles
    if rotate:
        for t in tiles:
            if t.xdim() < t.ydim():
                t.rot90()

    # calculate output image dimensions
    widths = []
    heights = []

    groups = np.array(groups)
    tiles = np.array(tiles)
    for t in np.unique(groups):
        if t in ignore:
            continue # skip ignored groups
        samples = np.argwhere(groups == t)[:, 0]
        widths.append(np.sum([t.xdim()+1 for t in tiles[samples]]))
        heights.append(np.max([t.ydim()+1 for t in tiles[samples]]))

    # create output arrays
    data = np.full((np.max(widths), np.sum(heights), tiles[0].band_count()), np.nan, dtype=tiles[0].data.dtype)
    label = np.full((np.max(widths), np.sum(heights), 1), 0, dtype=np.uint16)
    if ids is not None:
        ids = np.array(ids, dtype=np.uint16)
        sid = np.full((np.max(widths), np.sum(heights), 1), 0, dtype=np.uint16)

    # transfer data and labels
    _yoffs = 0
    bounds = []
    names = []
    for n in np.unique(groups):
        if n in ignore:
            continue # skip ignored groups
        samples = np.argwhere(groups == n)[:, 0]
        _xoffs = 0
        bb = [_xoffs,_yoffs]
        for i, _t,in enumerate(tiles[samples]):
            data[_xoffs:(_xoffs + _t.xdim()), _yoffs:(_yoffs + _t.ydim()), :] = _t.data
            label[_xoffs:(_xoffs + _t.xdim()), _yoffs:(_yoffs + _t.ydim()), 0] = n + 1
            if not ids is None:
                sid[_xoffs:(_xoffs + _t.xdim()), _yoffs:(_yoffs + _t.ydim()), 0] = ids[samples][i]

            _xoffs += _t.xdim()+1
        _yoffs += np.max([t.ydim() for t in tiles[samples]])+1
        bb += [_xoffs - bb[0], _yoffs - bb[1]]
        bounds.append(bb)
        names.append(n)

    # label areas with no data as background
    label[np.logical_not(np.isfinite(data).any(axis=-1))] = 0
    if ids is not None:
        sid[np.logical_not(np.isfinite(data).any(axis=-1))] = 0

    # make output images
    tiled_image = hylite.HyImage(data, header = tiles[0].header.copy())

    tiled_label = hylite.HyImage(label)
    tiled_label.set_band_names(["Class"])
    tiled_label.header['file type'] = 'ENVI Classification'
    tiled_label.header['class names'] = ['Background'] + ["Class %d" % (g+1) for g in names]
    tiled_label.header['classes'] = len(names) + 1

    if ids is not None:
        tiled_id = hylite.HyImage(sid)
        tiled_id.set_band_names(["ID"])
        tiled_id.header['file type'] = 'ENVI Classification'
        tiled_id.header['class names'] = ['Background'] + [str(i) for i in range(1, np.max(ids))]
        tiled_id.header['classes'] = np.max(ids) + 1

        return tiled_image, tiled_label, tiled_id, bounds

    return tiled_image, tiled_label, bounds

def build_core_template(images, N=5, thresh=40, vb=True):
    """
    Overlay images of core trays from e.g. a drillhole to calculate a template that is used for extracting
    individual core segments and is robust to data quirks (e.g. empty trays). All images must be identical
    dimensions and properly co-aligned.

    Args:
        images: a list of co-aligned images of different core trays build template with.
        N: the number of cores per tray. Default is 5.
        thresh: percentile used to separate foreground from background. Default is 40. Higher values ensure
                proper separation of cores, but will crop data more closely.
        vb: True if a tqdm progress bar should be printed.
    """

    # sum valid pixels
    valid = None
    loop = images
    if vb:
        loop = tqdm(images, leave=False, desc="Building template")
    for i in loop:
        if isinstance(i, str):  # load image if need be
            i = io.load(i)
        if valid is None:  # init valid if need be
            valid = np.zeros(i.data.shape[:-1])
        if not i is None:
            valid += np.isfinite(i.data).all(axis=-1)  # accumulate valid pixels

    # do threshold
    valid = valid > np.percentile(valid, thresh)
    if valid.shape[1] > valid.shape[0]:
        valid = valid.T

    # label components
    num_labels, labels_im = cv2.connectedComponents((valid.T > np.percentile(valid, 40)).astype(np.uint8))

    # take top N labels by area.
    area = [np.sum(labels_im == i) for i in range(num_labels)]
    thresh = np.sort(area)[::-1][N]
    l = 1
    for i in range(1, num_labels):
        if area[i] >= thresh:
            labels_im[labels_im == i] = l
            l += 1
        else:
            labels_im[labels_im == i] = 0  # background

    # return
    return hylite.HyImage(labels_im.T)

def unwrap_core(image, template, stack='linear', **kwds):
    """
    Unwrap drillholes in a core tray and stack them end to end along the x-axis, with the top at the left. Note that
    the top-left of the core tray is considered to be the "top", so the image may need to be flipped/mirrored prior
    to calling this function.

    Args:
        image: the image to unwrap. Background (non-core pixels) must have already been masked and set as np.nan.
        template: A labelled template
        stack: 'line' or 'separate'; either align cores in a single stick ('line'),
                or return individual tiles ('separate').
        **kwds: Keywords can include:

             - start = the meterage of the top of the core. Will be added to the image header.
             - end = the meterage of the bottom of the core. Will be added to the image header.
             - id = the id of this core tray.
    """

    # extract tiles
    tiles, ids = extract_tiles(image, template)

    # sort tiles into order
    idx = np.argsort(ids)
    tiles = np.array(tiles)[idx]

    # stack
    N = np.max(template.data)
    if 'sep' in stack.lower():

        # attribute tiles with position
        if 'start' in kwds and 'end' in kwds:
            l = (kwds['end'] - kwds['start']) / N  # length of a single stick of core
            start = np.linspace(kwds['start'], kwds['end'] - l, N)
            for i, t in enumerate(tiles):
                t.header['core start'] = start[i]  # start of each stick of core
                t.header['core end'] = start[i] + l  # end of each stick of core
        if 'id' in kwds:
            for i, t in enumerate(tiles):
                t.header['core id'] = kwds['id']

        # return
        return list(tiles)

    elif 'lin' in stack.lower():

        # attribute output with position
        out = image.copy()
        if 'start' in kwds:
            out.header['core start'] = kwds['start']
        if 'end' in kwds:
            out.header['core end'] = kwds['end']
        if 'id' in kwds:
            out.header['core id'] = kwds['id']

        # calculate width
        width = int(np.max([t.ydim() for t in tiles]))
        length = np.sum([t.xdim() for t in tiles])

        # reshape image
        data = np.full((length, width, image.band_count()), np.nan)
        x0 = 0
        for i, t in enumerate(tiles):
            data[x0:(x0 + t.xdim()), 0:t.ydim(), :] = t.data
            x0 += t.xdim()

        out.data = data
        return out
    else:
        assert False, "Error - %s is an unknown stacking." % stack

def tray_to_stick(tray, N):
    """
    Break a unwrapped tray into individual sticks.

    Args:
        tray: a HyImage instance containing a linearly unwrapped tray.
        N: the number of chunks of core to break it into.
    """

    # check orientation
    if tray.ydim() > tray.xdim():
        data = np.transpose(tray.data, (1, 0, 2))
    else:
        data = tray.data
    assert data.shape[0] >= data.shape[1], "Error - invalid shape? %s" % str(tray.data.shape)

    # slice
    step = int(data.shape[0] / N)
    slices = [data[(step * i):(step * (i + 1)), :, :] for i in range(N)]

    # build images
    start = float(tray.header['core start'])
    end = float(tray.header['core end'])
    zz = np.linspace(start, end, N + 1)
    images = []
    for i in range(N):
        image = hylite.HyImage(slices[i], header=tray.header.copy())
        image.header['core start'] = zz[i]
        image.header['core end'] = zz[i + 1]
        images.append(image)
    return images

def composite_cores(trays, pad=2):
    """
    Composites a list of uwrapped core trays (cf. unwrap_core(...)) into a single image such that each column
    represents a single tray (arranged vertically).

    Args:
        trays: a list of HyImage instances containing unwrapped core trays. The header of these
               images must contain localisation information; specifically the "core start" and "core end"
               keywords.
        pad: the padding (in pixels) between core segments.
    """

    # calculate order and identify breaks in core
    start = np.array([t.header['core start'] for t in trays])
    end = np.array([t.header['core end'] for t in trays])
    idx = np.argsort(start)
    breaks = np.argwhere(end[idx][:-1] - start[idx][1:] != 0)

    # rotate cores if need be
    for i, t in enumerate(trays):
        if t.xdim() > t.ydim():
            t = t.copy()
            t.data = np.transpose(t.data, (1, 0, 2))
            trays[i] = t

    # calculate dims and create output
    width = np.max([t.xdim() for t in trays]) + pad
    height = np.max([t.ydim() for t in trays])
    data = np.full((width * len(trays), height, trays[0].band_count()), np.nan)

    # transfer data
    x0 = 0
    y0 = 0
    px = [0]
    for i, n in enumerate(idx):
        core = trays[n].data
        x1 = x0 + core.shape[0]
        y1 = core.shape[1]
        data[x0:x1, y0:y1, :] = core
        x0 += width
        px.append(x0)

    # create output image
    out = hylite.HyImage(data)
    out.set_wavelengths(trays[0].get_wavelengths())
    out.header['wavelength units'] = trays[0].header.get('wavelength units', 'nm')
    out.header['core starts'] = start[idx]
    out.header['core ends'] = end[idx]
    out.header['core width'] = width  # width, in pixels, of individual core
    if len(breaks) > 0:
        out.header['core breaks'] = breaks

    return out

def plot_drillhole(composite, N=5, maxN=25, **kwds):
    """
    Plot a composite drillhole as created by composite_core. This plots the image data, but includes annotations
    the specify the distances along/down hole.

    Args:
        N: the number of sticks per tray. Used to ensure start/end labels are in the correct place.
        max_trays: the maximum number of trays per row. Default is 25.
    *Keywords*: are passed to HyImage.quick_plot( ... )
    """

    assert 'core width' in composite.header, "Error - no core width information in header file."
    assert 'core starts' in composite.header, "Error - no core position information in header file."
    assert 'core ends' in composite.header, "Error - no core position information in header file."

    # get data on dimensions and position
    width = int(composite.header['core width'])
    starts = composite.header['core starts']
    ends = composite.header['core ends']
    trayWidth = width * N  # width of one tray
    nTrays = int(composite.xdim() / trayWidth)

    if isinstance(starts, str):
        starts = np.fromstring(starts, sep=',')
    if isinstance(ends, str):
        ends = np.fromstring(ends, sep=',')

    # break into subimages
    nrows = int(np.ceil(nTrays / maxN))
    row_images = []
    for i in range(nrows):
        # copy data subset
        data = np.full((trayWidth * maxN, composite.ydim(), composite.band_count()), np.nan)
        x0 = i * trayWidth * maxN
        x1 = min((i + 1) * trayWidth * maxN, composite.xdim())
        data[0:(x1 - x0), :, :] = composite.data[x0:x1, :, :]

        # make new hyimage and store
        img = hylite.HyImage(data)
        if composite.has_wavelengths():
            img.set_wavelengths(composite.get_wavelengths())
        row_images.append(img)

    # make axes and plot sub-images
    dims = (18, nrows * 18 * row_images[0].ydim() / row_images[0].xdim())
    fig, ax = plt.subplots(nrows, 1, figsize=dims)
    ax = np.array(ax).ravel() # ensure ax is an array
    for i, a, im in zip(range(nrows), ax, row_images):
        im.quick_plot(ax=a, **kwds)
        a.set_yticks([])

        # set xticks
        _ends = np.array(ends[(N - 1 + i * maxN * N): min(N - 1 + (i + 1) * N * maxN, len(ends))][::N])
        _starts = np.array(starts[(i * maxN * N): min((i + 1) * N * maxN, len(starts))][::N])
        breaks = np.argwhere(_starts[1:] != _ends[:-1])  # calculate breaks where core trays don't line up

        # set ticks
        xticks = np.linspace(trayWidth, (len(_ends)) * trayWidth, len(_ends))
        a.set_xticks(xticks)
        a.set_xticklabels(["%.2f" % l for l in _ends])

        # add breaks
        # if len(breaks) > 0:
        #    for b in breaks[0]:
        #        a.axvline(xticks[b],color='r')

    fig.tight_layout()
    return fig, ax

def map_depth(image):
    # check we have the necessary info
    assert 'core width' in image.header, "Error - no core width information in header file."
    assert 'core starts' in image.header, "Error - no core position information in header file."
    assert 'core ends' in image.header, "Error - no core position information in header file."

    # get data on dimensions and position
    width = int(image.header['core width'])
    starts = image.header['core starts']
    ends = image.header['core ends']

    if isinstance(starts, str):
        starts = np.fromstring(starts, sep=',')
    if isinstance(ends, str):
        ends = np.fromstring(ends, sep=',')

    # build distance from top of each stick of core
    zz = np.zeros((image.data.shape[0], image.data.shape[1]))
    zz += np.linspace(0, 1, image.data.shape[1]) * (ends[0] - starts[0])

    # add value from previous core
    for i, x in enumerate(range(0, image.data.shape[0], width)):
        zz[x:min(x + width, image.data.shape[0])] += starts[i]

    # mask
    zz[np.logical_not(np.isfinite(image.data).any(axis=-1))] = np.nan

    return zz
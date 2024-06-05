"""
Load or import spectral libraries.
"""

import os
import numpy as np
import glob
import hylite
from hylite import HyLibrary
from hylite.io import makeDirs
from time import gmtime, strftime
from hylite.io.images import loadWithGDAL, saveWithGDAL, loadWithSPy, loadWithSPy
from pathlib import Path

# noinspection PyUnusedLocal
def _read_sed_file(path):

    """
    Read a (single) reference spectra from a .sed file.

    Args:
        path to the file.

    Returns:
        A tuple containing:

             - wav = a list of the wavelengths in the sed file
             - refl = a list of corresponding reflectances
             - name = a name for this sample (the file name without extension)
             - meta = a dictionary containing additional metadata in the header to the .sed file
     """

    with open(path, 'r') as f:
        # read header into metadata dict
        meta = {}
        l = f.readline()
        while not 'data:' in l.lower():
            splt = l[:-1].split(':')
            if len(splt) == 1:
                meta[splt[0]] = ""  # just store flag
            else:
                meta[splt[0]] = ''.join(splt[1:]).strip()
            l = f.readline()

        # get dataset name
        name = os.path.splitext(os.path.basename(path))[0]

        # read colum names
        # n.b. we ignore these for now. We could parse them by splitting by spaces EXCLUDING spaces
        # that are preceded by a '.' (i.e. split on ' ' but not '. ').
        col_names = f.readline()

        # read data lines
        wav = []
        refl = []
        l = f.readline()
        while l:
            # n.b. lines will be something like
            # 344.2	1.210000E+001	1.030000E+000	  7.815
            data = np.fromstring(l, sep='\t')
            wav.append(data[0])  # wavelength always first?
            refl.append(data[-1])  # reflectance always last?
            l = f.readline()
        # return
        return wav, refl, name, meta

def loadLibrarySED(path):
    """
    Load a spectral library stored as a folder of .sed files.

    Args:
        directory: the directory containing the .sed files. Filenames will be used as sample names.
    """

    assert os.path.isdir(path), "Error - must specify a directory, not a file."
    import utm

    # get sed files
    files = glob.glob(path + '/*.sed')
    assert len(files) > 0, "Error - no .sed files found in directory (%s)" % path

    # load sed files
    wav = None
    meta = None
    data = []
    names = []
    pos = []
    for f in files:
        _w, _r, _n, _m = _read_sed_file(f)  # read sed file

        # set or check wavelengths
        if wav is None:
            wav = _w
            meta = _m
        elif wav != _w:
            print("Error - .sed files have incompatable wavelength information")
            print("   Most recently read file has %d wavelengths. ", len(_w))
            print("   First file has %d wavelenghts.: ", len(wav))
            assert False

        data.append(_r)
        names.append(_n)

        if 'Latitude' in _m and 'Longitude' in _m and 'Altitude' in _m:
            try:
                _x = float(_m['Longitude'])
                _y = float(_m['Latitude'])
                _z = float(_m['Altitude'])

                # convert pos to UTM
                _x, _y, zn, zl = utm.from_latlon(_x, _y)
                meta['UTM Zone'] = '%d%s' % (zn, zl)
                pos.append([_x, _y, _z])
            except:
                pass  # not a float...

    # create spectral library
    library = HyLibrary(np.array(data), names, wav=np.array(wav))

    # add metadata to header
    ignore = ['comment', 'version', 'file name', 'columns [4]']
    for key, val in meta.items():
        if key.lower() in ignore:
            continue
        library.header[key] = val

    return library

def loadLibraryTSG(path):
    """
    Load a spectral library of TSG (The Spectral GeologistTM, e.g. HyLogger data) stored as a folder of .csv files.

    Args:
        directory: the directory containing the .sed files. Filenames will be used as sample names.
    """

    assert ".csv" in path, "Error - path must specify a csv file."

    # load TSG files
    meta = []
    refl = []
    pos = []

    # get dataset name
    name = os.path.splitext(os.path.basename(path))[0]

    with open(path, 'r') as f:
        # read wav
        wav_str = f.readline()
        wav = np.fromstring(wav_str.strip("Wavelength,"), sep=",")

        # read data lines
        l = f.readline()
        while l:
            # n.b. lines will be something like
            # NC3_0001_0009__T=1_L=1_P=9_D=6.0000003_X=559	0.0798395	0.0769894	0.0747408	0.0729613
            meta.append(l.split(",", 1)[0])
            refl.append(np.fromstring(l.split(",", 1)[1], sep=','))
            pos.append(float(((l.split(",", 1)[0]).split('D=' ,1)[1]).split('_X=' ,1)[0]))
            l = f.readline()
    names = [name +"_ " +str(x) for x in pos]

    # create spectral library
    library = HyLibrary(np.array(refl), names, wav=np.array(wav))
    library.header["depth"] = pos

    return library

def saveLibraryCSV(path, library):
    """
    Save a spectral library to a csv file with wavelengths as columns and sample reflectance as rows
    """

    # make directories if need be
    makeDirs(path)

    with open(path, 'w') as f:
        f.write( "name," + str(list(library.get_wavelengths()))[1:-1] +'\n' )
        for i,n in enumerate(library.get_sample_names()):
            f.write( str(n) + ", " + str(list( library.data[i,0,:] ))[1:-1] + '\n' )
    f.close()

def loadLibraryCSV(path):
    with open(path, 'r') as f:
        header = f.readline()
        wav = np.array( header.split(",")[1:], dtype=float)

        # parse spectra
        names = []
        refl = []
        l = f.readline()
        while l:
            l = l.split(",")
            names.append( l[0] )
            refl.append( np.array(l[1:],dtype=float) )
            l = f.readline()
    return HyLibrary( np.array(refl), names, wav=wav )


def loadLibraryTXT(path):
    """
    Load an ENVI text format library. This should have the following structure:

            ENVI ASCII Plot File
            Column 1: Wavelength
            Column 2: Sample1
            Column 3: Sample2
            C10 C20 C30
            C11 C21 C31
            C12 C22 C32
            ... ... ..

    """

    with open(path, 'r') as f:
        # check header line
        l0 = f.readline()
        assert 'envi ascii' in l0.lower(), "Error - Provided file is not an ENVI TXT library: %s" % path

        # read sample definitions
        ll = f.readline().lower()
        names = {}
        wav = -1
        i = 0
        while 'column' in ll:

            # split and get column number
            if ':' in ll:
                cname = ll.strip().split(':')[1]

                if 'wav' in cname:  # this column is wavelengths
                    wav = i
                else:  # this column is a spectra
                    names[i] = cname
                i += 1

            # read next line
            ll = f.readline().lower()

        # read data block
        data = []
        while ll:
            if ',' in ll:  # comma separated
                data.append(np.fromstring(ll, sep=','))
            else:  # space separated
                data.append(np.fromstring(ll, sep=' '))
            ll = f.readline()
        data = np.array(data).T
        f.close()  # close file

        # build hylibrary
        if wav > -1:
            wav = data[wav, :]
        else:
            wav = np.arange(data.shape[-1])
        lib = HyLibrary(data[list(names.keys()), :], lab=list(names.values()), wav=wav)
        return lib


import glob


def loadLibraryDIR(path, wav=None):
    """
    Load a spectral library from a directory of ENVI text files, with the following structure:

     - path
         - mineralA
             - spectraA.txt
             - spectraB.txt
             - spectraC.txt
         - mineralB
             -spectraA.txt
             -spectraB.txt
        etc.

    Data in this format can be downloaded using iSpec: https://www.samthiele.science/app/iSpec/index.html.


    Args:
        path: the directory path to search for ENVI spectra.
        wav: an array of wavelengths to resample the spectra onto. This is required as the txt files often
             have differing wavelength arrays. Data that does not overlap with wav will be set to nan. If None,
             this will be set to the wavelengths of the first encounted dataset.
    Returns:
        a HyLibrary spectral library instance.
    """

    files = glob.glob(str(Path(path)/ "*/*.txt"))
    libs = {}
    for f in files:
        # spectra name
        n = os.path.splitext(os.path.basename(f))[0]

        # mineral name
        m = os.path.dirname(f).split('/')[-1].split('\\')[-1]

        # load data
        lib = loadLibraryTXT(f)
        if wav is None:
            wav = lib.get_wavelengths()

        # check it is valid and resample to desired range
        delta = [np.min(np.abs(lib.get_wavelengths() - w)) for w in wav]
        if np.min(delta) < hylite.band_select_threshold:
            lib = lib.resample(wav, vb=False, partial=True)
            if m in libs:
                libs[m] = libs[m] + lib  # append
            else:
                libs[m] = lib

    # aggregate
    data = np.full((len(libs), np.max([l.sample_count() for l in libs.values()]), len(wav)), np.nan)
    lib = hylite.HyLibrary(data, lab=list(libs.keys()), wav=wav)
    for i, (k, v) in enumerate(libs.items()):
        lib.data[i, :v.data.shape[0], :] = v.data[:, 0, :]

    return lib

def saveLibraryTXT(path, library):
    """
    Save this library in envii TXT format (see loadLibraryTXT).
    """

    makeDirs(path)

    with open(path, 'w') as f:
        f.write("ENVI ASCII Plot File %s\n" % strftime("[%a %b %d %H:%M:%S %Y]", gmtime()))
        f.write('Column 1: Wavelength\n')

        # gather data and write column headers
        data = [library.get_wavelengths()]
        for i, n in enumerate(library.get_sample_names()):
            f.write('Column %d: %s %s\n' % (i + 2, n, n))
            data.append(np.nanmedian(library[n].data, axis=(0, 1)))  # we need to flatten samples to a single spectra

        # write data block
        for row in np.array(data).T:
            for v in row:
                f.write('  %.6f' % v)
            f.write('\n')
    f.close()

def saveLibraryLIB(path, library):
    path = os.path.splitext(path)[0] + ".lib" # ensure correct file format
    from hylite import io # N.B. this import must be here to avoid circular references
    io.save(path, library.as_image()) # default format is just as an image

def loadLibraryLIB(path):
    from hylite import io  # N.B. this import must be here to avoid circular references
    return io.load(path) # this is handled in the io.load function directly
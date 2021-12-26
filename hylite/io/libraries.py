import os
import numpy as np
import glob
from hylite import HyLibrary
from hylite.io import makeDirs
from hylite.io.images import loadWithGDAL, saveWithGDAL, loadWithSPy, loadWithSPy
# noinspection PyUnusedLocal
def _read_sed_file(path):

    """
    Read a (single) reference spectra from a .sed file.

    *Arguments*:
     - path to the file.

    *Returns*:
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

    *Arguments*
     - directory = the directory containing the .sed files. Filenames will be used as sample names.
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


    # sort out position info (convert to UTM)
    if not len(pos) == len(data):
        print(
            "Warning - Positions only defined for %d of %d spectra. Positions will be ignored." % (len(pos), len(data)))
        pos = None
    else:
        pos = np.array(pos)

    # create spectral library
    library = HyLibrary(names, np.array(data), wav=np.array(wav), pos=pos)

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

    *Arguments*
     - directory = the directory containing the .sed files. Filenames will be used as sample names.
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
    library = HyLibrary(names, np.array(refl), wav=np.array(wav))
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
            f.write( n + ", " + str(list( library.data[i,:] ))[1:-1] + '\n' )
    f.close()

def loadLibraryCSV(path):
    with open(path, 'r') as f:
        header = f.readline()
        wav = np.array( header.split(",")[1:], dtype=np.float)

        # parse spectra
        names = []
        refl = []
        l = f.readline()
        while l:
            l = l.split(",")
            names.append( l[0] )
            refl.append( np.array(l[1:],dtype=np.float) )
            l = f.readline()
    return HyLibrary( np.array(refl), names, wav=wav )

def saveLibraryLIB(path, library):
    path = os.path.splitext(path)[0] + ".lib" # ensure correct file format
    from hylite import io # N.B. this import must be here to avoid circular references
    io.save(path, library.as_image()) # default format is just as an image

def loadLibraryLIB(path):
    from hylite import io  # N.B. this import must be here to avoid circular references
    return io.load(path) # this is handled in the io.load function directly
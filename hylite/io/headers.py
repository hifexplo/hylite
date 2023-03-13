"""
Read ENVI header files.
"""

import os
import glob
import numpy as np
from hylite import HyHeader
import re

def makeDirs(path):
    """
    Checks if all the directories to the specified file exist, and if they do not, makes them.

    Args:
        path: the directory or file path to check. If any directories do not exist, they will be created.
    """

    # convert path to directory
    if not os.path.isdir(path):
        path = os.path.dirname(path)

    # make directory if need be
    if not os.path.exists(path) and path != '':
        os.makedirs(path)

def saveHeader(path, header):
    """
    Write to a header file.

    Args:
        path: the path to write to.
        header: the HyHeader object to write.
    """

    if os.path.exists(path):
        os.remove(path)  # delete existing header so we can overwrite it (causes weird bug if we don't....)

    with open( os.open( path, os.O_CREAT | os.O_WRONLY, 0o777) , "w+") as f: # create file with appropriate permissions
        f.write("ENVI\n")
        for key, value in header.items():

            #get value as string
            if isinstance(value, list) or isinstance(value, np.ndarray): #convert lists to string
                if isinstance(value[0], str):  # format a list of strings
                    s = ', '.join(value)
                else:  # format a list of numbers
                    s = ("%s" % list(value))[1:-1]  # convert to string and strip '[' and ']' characters
            else:
                s = str(value).strip()

            #write in the relevant format
            if s == "":
                f.write("%s = { }\n" % key)
            elif "," in s:
                if len(s) < 512:
                    f.write("%s = {%s}\n" % (key, s))
                else:
                    f.write("%s = {" % key)
                    splt = s.split(",")
                    for i, l in enumerate(splt[:-1]):
                        if i % 10 == 0: # new line every 10th value
                            f.write("\n")
                        f.write("%s," % l) # write values
                    f.write("%s}\n" % splt[-1])
            else:
                f.write("%s = %s\n" % (key, s))
        f.close()

def loadHeader(path):
    """
    Load a header file.

    Args:
       file: a file path to a .hdr file.
    """
    header = HyHeader()

    # store path
    header['path'] = path

    inblock = False
    try:
        hdrfile = open(path, "r")
    except:
        if path is not None: # no header file exists. This is not necessarily an issue.
            print("Could not open hdr file %s" % str(path))
        return None

    # Read line, split it on equals, strip whitespace from resulting strings and add key/value pair to output
    currentline = hdrfile.readline()
    while (currentline != ""):
        # ENVI headers accept blocks bracketed by curly braces - check for these
        if not inblock:
            # Split line on first equals sign
            if (re.search("=", currentline) is not None):
                linesplit = re.split("=", currentline, 1)
                # key = str.lower(linesplit[0].strip())
                key = linesplit[0].strip()
                value = linesplit[1].strip()

                # If value starts with an open brace, it's the start of a block - strip the brace off and read the rest of the block
                if (re.match("{", value) is not None):
                    inblock = True
                    value = re.sub("^{", "", value, 1)
                    # If value ends with a close brace it's the end of the block as well - strip the brace off
                    if (re.search("}$", value)):
                        inblock = False
                        value = re.sub("}$", "", value, 1)
                value = value.strip()
                header[key] = value
        else:

            # If we're in a block, just read the line, strip whitespace (and any closing brace ending the block) and add the whole thing
            value = currentline.strip()
            if (re.search("}$", value)):
                inblock = False
                value = re.sub("}$", "", value, 1)
                value = value.strip()
            header[key] = header[key] + value
        currentline = hdrfile.readline()
    hdrfile.close()

    #convert default things like wavelength data to numeric form
    #N.B. wavelength should ALWAYS be stored as nanometres
    if 'Wavelength' in header: # drop upper case wavelength for some files
        header['wavelength'] = header['Wavelength']
        del header['Wavelength']
    if "wavelength" in header:
        units = header.get("wavelength units", "nm").lower()
        if "nm" in units or "nano" in units: #units in nanometers
            header['wavelength'] = np.fromstring(header['wavelength'], sep=',')
        elif "um" in units or "micro" in units:
            header['wavelength'] = np.fromstring(header['wavelength'], sep=',') * 1000.0
        elif "mm" in units or "milli" in units:
            header['wavelength'] = np.fromstring(header['wavelength'], sep=',') * 1000000.0
        elif "wavenum" in units or "cm-1" in units:
            header['wavelength'] = (1 / np.fromstring(header['wavelength'], sep=',')) * 10000000.0
        elif "unk" in units.lower():
            print("Warning - unknown wavelength units. Assuming nanometers.")
            header['wavelength'] = np.fromstring(header['wavelength'], sep=',')
        else:
            assert False, "Error - unrecognised wavelength format %s." % units

        header['wavelength units'] = 'nm' #update wavelength units

    #if "band names" in header:
    #    header["band names"] = header['band names'].split(',') #split into list

    if "fwhm" in header:
        header["fwhm"] = np.fromstring(header['fwhm'], sep=',') #split into list

    if "bbl" in header:
        header["bbl"] = np.fromstring(header['bbl'], sep=',') #split into list

    return header

def matchHeader(path):
    """
    Matches image and header path.

    Args:
        path: the path to an image or header file

    Returns:
        A tuple containing:

        - header = file path to the associated .hdr or .HDR file (if found, otherwise None)
        - image = file path to the associated image data (if found, otherwise None).
    """

    # find files with the same name but different extensions
    path, ext = os.path.splitext(path)
    header = None
    image = None
    match = glob.glob(path + "*")

    assert (path + ext) in match, "Error - file not found (%s)" % (path + ext)
    match.remove(path + ext)  # remove self-match

    # we have a header file, find associated image
    if "hdr" in str.lower(ext):
        header = path + ext  # store header file

        # did we find image data?
        for m in match:
            # ignore potentially associated file types (that aren't the data file we're looking for)
            if os.path.splitext(m)[0] == path \
                    and not "log" in str.lower(os.path.splitext(m)[1]) \
                    and not "png" in str.lower(os.path.splitext(m)[1]) \
                    and not "jpg" in str.lower(os.path.splitext(m)[1]) \
                    and not "bmp" in str.lower(os.path.splitext(m)[1]) \
                    and not "hdt" in str.lower(os.path.splitext(m)[1]) \
                    and not "csv" in str.lower(os.path.splitext(m)[1]) \
                    and not "txt" in str.lower(os.path.splitext(m)[1]) \
                    and not "xml" in str.lower(os.path.splitext(m)[1]) \
                    and not "cam" in str.lower(os.path.splitext(m)[1]) \
                    and not "brm" in str.lower(os.path.splitext(m)[1]):

                image = m  # store matching image file
                break
            elif (os.path.splitext(m)[0] == path) and (image is None):
                image = m # .png files can be valid, but should be lower priority, so store but continue looking
                          # in case there is e.g., a .dat file also.
    # we have an image file, find associated header file
    else:
        image = path + ext
        for m in match:
            if ".hdr" in str.lower(m) and os.path.splitext(m)[0] == path:
                header = m
                break

    return header, image


"""
Simplify file input and output using HyCollection directory structures.
"""

import os
import hylite
import numpy as np
import shutil
import re
from natsort import natsorted
class External(object):
    """
    Small wrapper class for storing external objects in HyCollections.
    """
    def __init__( self, path, base=None ):
        self.path = path
        self.base = base
        self.value = None

    def get(self):

        # is attr already loaded?
        if self.value is not None:
            return self.value

        # no - figure out where it is
        if self.base is not None:
            path = os.path.join(self.base, self.path)
        assert os.path.exists(path), "Error: %s does not exist." % path

        # and load it
        try:
            self.value = hylite.io.load( path )
            return self.value
        except:
            assert False, "Error loading external attribute %s" % path

class HyCollection(object):
    """
    A utility class for mapping data stored in a special file system (.hyc directory or similar) between RAM and disk storage.
    Useful for (1) reducing IO code and (2) writing out-of-core analyses. The underlying .hyc directory can contain any
    data that can be read and written by hylite.io, including: numpy arrays, numbers, strings and hylite.HyData instances.
    """

    def __init__(self, name, root, header=None, vb=False):
        """
        Args:
            name (str): a name for this HyCollection. Names of HyCollections stored in any given directory must be
                  be unique to avoid conflicts.
            root (str): the location of this HyCollection on disk.
            header (hylite.HyHeader): a header file for this HyCollection. If None (default) a new header will be created.
            vb (bool): True if print notifications should be written when data is being loaded from disk. Default is False.
        """
        self.name = os.path.splitext(name)[0]  # trim extension just in case
        self.root = root
        if header is None:
            header = hylite.HyHeader()
        header['file type'] = 'Hylite Collection'  # ensure file type is correct (just in case someone cares)
        self.header = header
        self.vb = vb
        self.ext = '.hyc'

    def get_file_dictionary(self, root=None, name=None):
        """
        Convert this object to a dictionary of files (keys) and serializable objects (values).
        Note that primitive attributes (string, integer, etc.) will be stored in the header file.

        Args:
            root (str): the directory to store this HyCollection in. Defaults to the root directory specified when
                  this HyCollection was initialised, but this can be overriden for e.g. saving in a new location.
            name (str): the name to use for the HyCollection in the file dictionary. If None (default) then this instance's
                  name will be used, but this can be overriden for e.g. saving in a new location.
        Returns:
            a dictionary such that dict[ path ] = object.
        """
        # parse root
        if root is None:
            root = self.root
        if name is None:
            name = self.name
        assert root is not None, "Error - root argument must be set during HyCollection initialisation or function call."
        assert name is not None, "Error - name argument must be set during HyCollection initialisation or function call."

        # get all attributes (excluding class methods/variables and the header variable)
        attr = self.getAttributes()

        # build paths dictionary
        out = {os.path.join(root, "%s.hdr" % name): self.header}
        path = self.getDirectory(root=root, name=name)
        for a in attr:
            value = getattr(self, a)  # get value
            if value is None:
                continue # skip None values
            elif type(value) in [int, str, bool, float]:  # primitive types
                self.header[a] = value  # store in header file
            elif isinstance(value, External): # external link
                self.header[a] = "<" + value.path + ">" # store in header file
            else:
                out[os.path.join(path, a)] = value

        return out

    def clean(self):
        """
        Delete files associated with attributes that have been cleared by setting to None.
        """
        attr = self.getAttributes()
        for a in attr:
            value = getattr(self, a)
            if value is None:
                # remove from header file? (easy)
                if a in self.header:
                    del self.header[a]
                else:
                    # remove from disk...
                    # solve path from attribute name
                    path = None
                    for f in os.listdir(self.getDirectory()):
                        if os.path.splitext(f)[0] == a:  # we have found the right file
                            path = os.path.join(self.getDirectory(), f)
                            break
                    if path is not None and os.path.exists(path):
                        hdr, dat = hylite.io.matchHeader( path )
                        if hdr is not None and os.path.exists(hdr):
                            os.remove(hdr)
                        if dat is not None and os.path.exists(dat) and os.path.isdir(dat): # nested HyCollection
                            shutil.rmtree(dat)
                        if os.path.exists(dat) and os.path.isfile(dat): # other data type
                            os.remove(dat)
                # remove attribute
                delattr(self, a)

    def _loadAttribute_(self, attr):
        """
        Load an attribute from disk. This should generally not be called directly - rather it is called
        if needed when any class attributes are requested.

        Args:
            attr (str): the attribute name to load.
        """

        # check if attribute is in the header file
        attr = attr.strip().replace(' ', '_')

        # edge case - the attribute being requested is the header file!
        if 'header' == attr or '__' in attr:  # ignore headers and private (__x__) variables.
            raise AttributeError(attr)

        if (attr in self.header) or (attr.replace('_', ' ') in self.header):
            # get value from header file
            if attr in self.header:
                val = self.header[attr] # no spaces
            else:
                val = self.header[attr.replace('_', ' ')] # possibly has spaces

            # parse strings if needed
            if isinstance(val,str):
                # load as a list
                if '{' in val and '}' in val:
                    val = self.header.get_list(val)
                elif '<' in val and '>' in val: # load as an external path
                    val = self.header[val].strip()[1:-1]
                    if os.path.exists(val): # absolute path!
                        val = External( val, None) # wrap in absolute External class
                    else:
                        val = External( val, self.root ) # wrap in relative External class
                else:  # is it an integer or a float?
                    try:
                        val = int(val)
                    except:
                        try:
                            val = float(val)
                        except:
                            # finally, try converting boolean types
                            if val.lower() == 'true':
                                val = True
                            elif val.lower() == 'false':
                                val = False

            # done
            self.__setattr__(attr, val)  # easy!

        # attribute not in header; must be loaded from disk.
        else:
            # no file associated with this HyCollection - raise attribute error.
            if not os.path.exists(self.getDirectory(makedirs=False)):
                raise AttributeError

            # solve path from attribute name
            path = None
            for f in os.listdir(self.getDirectory(makedirs=False)):
                if os.path.splitext(f)[0] == attr:  # we have found the right file
                    path = os.path.join(self.getDirectory(makedirs=False), f)
                    break
            assert path is not None and os.path.exists(path), \
                "Error - could not load attribute %s from disk (%s)." % ( attr, self.getDirectory(makedirs=False))

            # load attribute
            if self.vb:
                print("Loading %s from %s" % (attr, path))
            self.__setattr__(attr, hylite.io.load(path))  # load and update HyCollection attribute

    def get_path(self, name: str):
        """
        Return the path of the specified attribute. Note that this file may or may not exist, depending
        on if this HyCollection has been saved previously. Also note that this path will exclude the file extension.
        """
        if name == 'header' or name in self.header:
            return os.path.splitext( self.getDirectory() )[0]
        else:
            name = name.strip().replace(' ', '_')
            return os.path.join( self.getDirectory(), name )

    def getDirectory(self, root=None, name=None, makedirs=False):
        """
        Return the directory files associated with the HyCollection are stored in.

        Args:
            root (str): the directory to store this HyCollection in. Defaults to the root directory specified when
                  this HyCollection was initialised, but this can be overriden for e.g. saving in a new location.
            name (str): the name to use for the HyCollection in the file dictionary. If None (default) then this instance's
                  name will be used, but this can be overriden for e.g. saving in a new location.
            makedirs (bool): True if this directory should be created if it doesn't exist. Default is False.
        """
        if root is None:
            root = self.root
        if name is None:
            name = self.name
        assert root is not None, "Error - root argument must be set during HyCollection initialisation or function call."
        assert name is not None, "Error - name argument must be set during HyCollection initialisation or function call."
        p = os.path.join(root, os.path.splitext(name)[0] + self.ext)
        if makedirs:
            os.makedirs(p, exist_ok=True)  # ensure directory actually exists!
        return p

    def getAttributes(self, ram_only=True, file_formats=False):
        """
        Return a list of available attributes in this HyCollection.

        Args:
         - ram = True if only attributes loaded in RAM should be included. Default is True.
         - file_formats = True if the file extensions of attributes stored on disk should be retained. Default is False.
        """
        # get potential attributes
        attr = list(set(dir(self)) - set(dir(HyCollection)) - set(['header', 'root', 'file type','name', 'ext', 'vb']))

        # loop through and remove all functions
        out = []
        known = []
        for a in attr:
            if '__' in a:
                continue # ignore private variables
            if not type(super().__getattribute__(a)) == type(self.getAttributes): # ignore methods
                if file_formats:
                    out.append((a,type(self.get(a)).__name__))
                else:
                    out.append(a)
                known.append(a)

        # add elements in header
        for a,v in self.header.items():
            if a not in known:
                if file_formats:
                    if file_formats:
                        out.append((a, type(self.get(a)).__name__))
                    else:
                        out.append(a)
                    known.append(a)

        # also add attributes on disk
        if not ram_only:
            if os.path.exists(self.getDirectory()):
                for f in os.listdir(self.getDirectory()):
                    name, ext = os.path.splitext(f)
                    if name not in known and ext != '.hdr' and ext != '':
                        if file_formats:
                            out.append((name,ext))
                        else:
                            out.append(name)
                        known.append(name)
        return out

    def query(self, *, name_pattern=None, ext_pattern=None, recurse=False, recurse_matches=False, ram_only=False):
        """
        Finds attributes of this HyCollection with names or types matching the specified patterns. Note that (1)
        if both name_pattern and ext_pattern are provided then attributes must match both filters to be included in the
        results, and (2)

        Args:
         - name_pattern (list, str) = A regex pattern (string) or list of regex pattern strings to match against. If an attribute name
                          matches against any of the provided patterns then it will be included in the output.
                          Default is None (match all attributes).
         - ext_pattern (list, str) = A regex pattern (string) or list of regex pattern strings to match against. Matches will be evaluated
                         against file extensions for attributes on the disk (e.g., ".hdr") and type names (e.g., "HyImage") for
                         attributes loaded in RAM (as we cannot guess what their file extension may be). Note that class inheritance
                         is not considered during this matching, so e.g., "HyData" will not match with "HyImage".
                         Default is None (match all attributes).
         - recurse (bool) = True if (all) child HyCollections should also be queried to search the entire HyCollection tree for
                     matches. Default is False.
         - recurse_matches (bool) = True if HyCollections that match the provided filters should also be queried recursively. Default
                    is False.
         - ram_only (bool) = True if only attributes already loaded into memory should be queried. Default is False.
        """

        def match(value, pattern):
            """
            Quick function for matching against list or regex.
            """
            if pattern is None:
                return True
            if not isinstance(pattern, list) or isinstance(pattern, tuple) or isinstance(pattern, np.ndarray):
                pattern = [pattern]
            for p in pattern:
                assert isinstance(p,str), "Error - %s is an invalid pattern [ should be string ]" % p
                if re.search(p, value) is not None:
                    return True  # we have a match!
            return False

        attr = self.getAttributes(ram_only=ram_only, file_formats=True)
        out = []
        for a,e in attr:

            # do matching
            if match(a, name_pattern) and match(e, ext_pattern):
                out.append(a)
                if not recurse_matches:
                    continue # skip to next loop

            # recurse if required
            if recurse:
                if self.loaded(a): # object already loaded in ram
                    if isinstance( self.get(a), hylite.HyCollection ):
                        out += self.get(a).query( name_pattern=name_pattern, ext_pattern=ext_pattern,
                                                  recurse=recurse, recurse_matches=recurse_matches,
                                                  ram_only = ram_only )
                else: # object is on the disk
                    from hylite.io import _loadCollection
                    try:
                        C = _loadCollection(os.path.join(self.getDirectory(), a + e) )
                        out += C.query( name_pattern=name_pattern, ext_pattern=ext_pattern,
                                                      recurse=recurse, recurse_matches=recurse_matches,
                                                      ram_only = ram_only )
                    except:
                        pass # continue, this was not a HyCollection
        return natsorted(out) # sort alphabetically for consistency

    def loaded(self, name):
        """
        Return True if the requested attribute is loaded into RAM already, and False if it exists on the disk.
        Throws an attribute error if the attribute does not exist.
        """
        try:  # try getting attribute
            attr = object.__getattribute__(self, name.strip().replace(' ','_'))
            return True
        except:
            files = self.getAttributes(ram_only=False)
            assert name in files, "Error - attribute %s does not exist." % name
            return False

    def print(self):
        """
        Print a nicely formatted summary of the contents of this collection.
        """
        attr = self.getAttributes()

        # print loaded variables
        print("Attributes stored in RAM:")
        for a in attr:
            v = getattr(self, a)
            print("\t - %s called %s" % (type(v), a) )

        # print header variables
        print("Attributes stored in header:")
        for k,v in self.header.items():
            if k not in ['file type','file_type', 'path'] and k not in attr: # header keys to ignore
                if isinstance(v,str) or isinstance(v, list) or isinstance(v, np.ndarray):
                    print("\t %s = %s"% (k,type(v))) # print type
                else:
                    print("\t %s = %s" % (k, v)) # print value

        # print disk variables
        if os.path.exists(self.getDirectory()):
            print("Attributes stored on disk:")
            for f in os.listdir(self.getDirectory()):
                name, ext = os.path.splitext(f)
                if name not in attr and ext != '.hdr':
                    print( "\t - %s" % f)

    def save(self):
        """
        Quick utility function for saving this in the predefined location.
        """
        from hylite import io # occasionally io doesn't seem to get loaded unless we call this ... strange?
        hylite.io.save(os.path.splitext(self.getDirectory())[0], self)

    def save_attr(self, attr):
        """
        Save a single attribute in this HyCollection.
        """
        from hylite import io  # occasionally io doesn't seem to get loaded unless we call this ... strange?
        if attr in self.header:
            hylite.io.saveHeader( os.path.splitext(self.getDirectory())[0] + '.hdr', self.header )
        else:
            hylite.io.save( self.get_path(attr), self.get(attr) )

    def free(self):
        """
        Free all attributes in RAM. To avoid losing data, be sure to save this HyCollection first (e.g. using
        self.save(...).
        """
        attr = self.getAttributes()
        for a in attr:
            delattr(self, a)

    def free_attr(self, a):
        """
        Unload the specific attribute from RAM. Can be used to e.g. remove unchanged attributes before saving. Note that any unsaved
        changes to this attribute will be lost.
        """
        delattr(self, a)

    def addExternal(self, name, path, relative=True):
        """
        Add an external link (that is saved/loaded by this HyCollection instance, but not stored in its data folder).

        Args:
         - name: the name of the attribute to add.
         - path: the path to the object to add.
         - relative: True if the path should be converted to a relative one. Default is True.
        """

        assert os.path.exists(path), "Error - %s is not a valid file or folder." % path

        if relative:
            path = os.path.relpath( path, self.root )
            self.__setattr__(name, External(path,self.root))
        else:
            self.__setattr__(name, External(path, None))

    def addSub(self, name):
        """
        Add a subcollection to this one (and sort out internal file paths etc.).

        Args:
            name (str): the name of the subcollection to add.

        Returns:
            a HyCollection object representing the subcollection.
        """

        if self.root is None: # no path defined
            S = HyCollection(name,'')
        else:
            S = HyCollection(name, self.getDirectory(makedirs=False))
        self.__setattr__(name, S)
        return S

    # expose getters and setters for easy access
    def get(self, name):
        """
        Get an attribute in this collection.
        """
        return self.__getattribute__(name)



    def set(self, name, value, save=False):
        """
        Set an attribute in this collection.

        Args:
         - name = the name of the variable to set.
         - value = the value to set this variable too.
         - save = True if the variable should immediately be saved to disk.
        """
        self.__setattr__(name, value)
        if save:
            self.save_attr(name)

    def __getattribute__(self, name):
        """
        Override __getattribute__ to automatically load out-of-core attributes if they are asked for.
        """
        # attribute has not yet been loaded - do so now
        try:  # try getting attribute
            attr = object.__getattribute__(self, name.strip().replace(' ', '_'))
        except AttributeError:  # no attribute found
            self._loadAttribute_(name)  # load the attribute from disk
            if not hasattr(self, name):
                raise AttributeError(name)
            else:
                attr = object.__getattribute__(self, name.strip().replace(' ', '_') )

        # resolve external links if necessary
        if isinstance(attr, External):
            return attr.get()

        # return
        return attr

    def __setattr__(self, name, value):
        """
        Override __setattr__ to throw an error when dodgy data types are added to this collection. Note that private
        attributes (with __ in their name, such as __foo__) are allowed to have any type, but will not be written to
        disk.
        """
        if '__' not in name:
            valid = type(value) in [ int, str, bool, float, list ] # accept primitive types
            valid = valid or isinstance( value, np.ndarray) # accept numpy arrays
            valid = valid or isinstance( value, hylite.HyData ) # accept hydata types
            valid = valid or isinstance(value, hylite.HyHeader)  # accept hydata types
            valid = valid or isinstance( value, hylite.HyCollection ) # accept HyCollection instances (nesting)
            valid = valid or isinstance(value, hylite.project.Camera ) # accept Camera instances
            valid = valid or isinstance(value, hylite.project.Pushbroom)  # accept Pushbroom instances
            valid = valid or isinstance(value, hylite.project.PMap)  # accept Pushbroom instances
            valid = valid or isinstance(value, External ) # accept external links
            valid = valid or isinstance(value, list) and np.array( [isinstance(d, hylite.HyData) ] for d in value ).all() # multimwl maps
            valid = valid or value is None # also accept None
            assert valid, "Error - %s is an invalid attribute type for HyCollection." % type(value)
        object.__setattr__(self, name.strip().replace(' ', '_'), value)
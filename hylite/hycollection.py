import os
import hylite
import numpy as np

class HyCollection(object):

    def __init__(self, name, root, header=None, vb=False):
        """
        Create a new hycollection instance.

        *Arguments*:
         - name = a name for this HyCollection. Names of HyCollections stored in any given directory must be
                  be unique to avoid conflicts.
         - root = the location of this HyCollection on disk.
         - header = a header file for this HyCollection. If None (default) a new header will be created.
         - vb = True if print notifications should be written when data is being loaded from disk. Default is False
                (or what is specified by a vb key in the header file).
        """
        self.name = os.path.splitext(name)[0]  # trim extension just in case
        self.root = root
        if header is None:
            header = hylite.HyHeader()
        header['file type'] = 'Hylite Collection'  # ensure file type is correct (just in case someone cares)
        self.header = header
        if 'vb' not in header:
            self.vb = vb

    def get_file_dictionary(self, root=None):
        """
        Convert this object to a dictionary of files (keys) and serializable objects (values).
        Note that primitive attributes (string, integer, etc.) will be stored in the header file.

        *Arguments*:
         - root = the directory to store this HyCollection in. Defaults to the root directory specified when
                  this HyCollection was initialised, but this can be overriden for e.g. saving in a new location.
        *Returns*:
         - a dictionary such that dict[ path ] = object.
        """
        # parse root
        if root is None:
            root = self.root
        assert root is not None, "Error - root argument must be set during HyCollection initialisation or function call."

        # get all attributes (excluding class methods/variables and the header variable)
        attr = list(set(dir(self)) - set(dir(HyCollection)) - set(['header', 'root']))

        # build paths dictionary
        out = {os.path.join(root, "%s.hdr" % self.name): self.header}
        path = os.path.join(root, self.name) + ".hyc"
        for a in attr:

            value = getattr(self, a)  # get value
            if type(value) in [int, str, bool, float]:  # primitive types
                self.header[a] = value  # store in header file
            else:
                out[os.path.join(path, a)] = value

        return out

    def _loadAttribute_(self, attr):
        """
        Load an attribute from disk. This should generally not be called directly - rather it is called
        if needed when any class attributes are requested.

        *Attr*:
         - the attribute name to load.
        """

        # edge case - the attribute being requested is the header file!
        if 'header' == attr or '__' in attr:  # ignore headers and private (__x__) variables.
            return

        # no file associated with this HyCollection - raise attribute error.
        if not os.path.exists(os.path.join(self.root, self.name) + '.hyc'):
            raise AttributeError

        # check if attribute is in the header file
        if attr in self.header:
            # get value from header file
            val = self.header[attr]

            # load as a list
            if '{' in val and '}' in val:
                val = self.header.get_list(val)
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
            # solve path from attribute name
            path = None
            for f in os.listdir(os.path.join(self.root, self.name + ".hyc")):
                if os.path.splitext(f)[0] == attr:  # we have found the right file
                    path = os.path.join(os.path.join(self.root, self.name + ".hyc"), f)
                    break
            assert path is not None and os.path.exists(path), \
                "Error - could not load attribute %s from disk (%s)." % (
                attr, os.path.join(self.root, self.name + ".hyc/"))

            # load attribute
            if self.vb:
                print("Loading %s from %s" % (attr, path))
            self.__setattr__(attr, hylite.io.load(path))  # load and update HyCollection attribute

    def __getattribute__(self, name):
        """
        Override __getattribute__ to automatically load out-of-core attributes if they are asked for.
        """
        # attribute has not yet been loaded - do so now
        try:  # try getting attribute
            return object.__getattribute__(self, name)
        except AttributeError:  # no attribute found
            self._loadAttribute_(name)  # load the attribute from disk
            return object.__getattribute__(self, name)  # return it

    def __setattr__(self, name, value):
        """
        Override __setattr__ to throw an error when dodgy data types are added to this collection.
        """
        valid = type(value) in [ int, str, bool, float, list ] # accept primitive types
        valid = valid or isinstance( value, np.ndarray) # accept numpy arrays
        valid = valid or isinstance( value, hylite.HyData ) # accept hydata types
        valid = valid or isinstance(value, hylite.HyHeader)  # accept hydata types
        valid = valid or isinstance( value, hylite.HyCollection ) # accept HyCollection instances (nesting)
        valid = valid or isinstance(value, hylite.project.Camera ) # accept Camera instances
        valid = valid or isinstance(value, hylite.project.Pushbroom)  # accept Pushbroom instances
        assert valid, "Error - %s is an invalid attribute type for HyCollection." % type(value)
        object.__setattr__(self, name, value)
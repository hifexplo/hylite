import os
import hylite
import numpy as np
import shutil

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

    def __init__(self, name, root, header=None, vb=False):
        """
        Create a new HyCollection instance.

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

    def get_file_dictionary(self, root=None, name=None):
        """
        Convert this object to a dictionary of files (keys) and serializable objects (values).
        Note that primitive attributes (string, integer, etc.) will be stored in the header file.

        *Arguments*:
         - root = the directory to store this HyCollection in. Defaults to the root directory specified when
                  this HyCollection was initialised, but this can be overriden for e.g. saving in a new location.
         - name = the name to use for the HyCollection in the file dictionary. If None (default) then this instance's
                  name will be used, but this can be overriden for e.g. saving in a new location.
        *Returns*:
         - a dictionary such that dict[ path ] = object.
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
        path = self._getDirectory( root=root, name=name )
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
                    for f in os.listdir(self._getDirectory()):
                        if os.path.splitext(f)[0] == a:  # we have found the right file
                            path = os.path.join(self._getDirectory(), f)
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

        *Attr*:
         - the attribute name to load.
        """

        # edge case - the attribute being requested is the header file!
        if 'header' == attr or '__' in attr:  # ignore headers and private (__x__) variables.
            return

        # no file associated with this HyCollection - raise attribute error.
        if not os.path.exists( self._getDirectory() ):
            raise AttributeError

        # check if attribute is in the header file
        if attr in self.header:
            # get value from header file
            val = self.header[attr]

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
            # solve path from attribute name
            path = None
            for f in os.listdir( self._getDirectory() ):
                if os.path.splitext(f)[0] == attr:  # we have found the right file
                    path = os.path.join( self._getDirectory(), f)
                    break
            assert path is not None and os.path.exists(path), \
                "Error - could not load attribute %s from disk (%s)." % ( attr, self._getDirectory() )

            # load attribute
            if self.vb:
                print("Loading %s from %s" % (attr, path))
            self.__setattr__(attr, hylite.io.load(path))  # load and update HyCollection attribute

    def _getDirectory(self, root=None, name=None):
        """
        Return the directory files associated with the HyCollection are stored in.

         *Arguments*:
         - root = the directory to store this HyCollection in. Defaults to the root directory specified when
                  this HyCollection was initialised, but this can be overriden for e.g. saving in a new location.
         - name = the name to use for the HyCollection in the file dictionary. If None (default) then this instance's
                  name will be used, but this can be overriden for e.g. saving in a new location.
        """
        if root is None:
            root = self.root
        if name is None:
            name = self.name
        assert root is not None, "Error - root argument must be set during HyCollection initialisation or function call."
        assert name is not None, "Error - name argument must be set during HyCollection initialisation or function call."

        return os.path.join(root, os.path.splitext(name)[0] + ".hyc")

    def getAttributes(self):
        """
        Return a list of available attributes in this HyCollection.
        """
        # get potential attributes
        attr = list(set(dir(self)) - set(dir(HyCollection)) - set(['header', 'root', 'name']))

        # loop through and remove all functions
        out = []
        for a in attr:
            if not type(super().__getattribute__(a)) == type(self.getAttributes): # ignore methods
                out.append(a)
        return out

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
            if k not in ['file type', 'path'] and k not in attr: # header keys to ignore
                if isinstance(v,str) or isinstance(v, list) or isinstance(v, np.ndarray):
                    print("\t %s = %s"% (k,type(v))) # print type
                else:
                    print("\t %s = %s" % (k, v)) # print value

        # print disk variables
        if os.path.exists(self._getDirectory()):
            print("Attributes stored on disk:")
            for f in os.listdir(self._getDirectory()):
                name, ext = os.path.splitext(f)
                if name not in attr and ext != '.hdr':
                    print( "\t - %s" % f)

    def save(self):
        """
        Quick utility function for saving this in the predefined location.
        """
        from hylite import io # occasionally io doesn't seem to get loaded unless we call this ... strange?
        hylite.io.save( os.path.splitext( self._getDirectory() )[0], self )

    def free(self):
        """
        Free all attributes in RAM. To avoid losing data, be sure to save this HyCollection first (e.g. using
        self.save(...).
        """
        attr = self.getAttributes()
        for a in attr:
            delattr(self, a)

    def addExternal(self, name, path, relative=True):
        """
        Add an external link (that is saved/loaded by this HyCollection instance, but not stored in its data folder).

        *Arguments*:
         - name = the name of the attribute to add.
         - path = the path to the object to add.
         - relative = True if the path should be converted to a relative one. Default is True.
        """

        assert os.path.exists(path), "Error - %s is not a valid file or folder." % path

        if relative:
            path = os.path.relpath( path, self.root )
            self.__setattr__(name, External(path,self.root))
        else:
            self.__setattr__(name, External(path, None))

    def __getattribute__(self, name):
        """
        Override __getattribute__ to automatically load out-of-core attributes if they are asked for.
        """
        # attribute has not yet been loaded - do so now
        try:  # try getting attribute
            attr = object.__getattribute__(self, name)
        except AttributeError:  # no attribute found
            self._loadAttribute_(name)  # load the attribute from disk
            attr = object.__getattribute__(self, name)

        # resolve external links if necessary
        if isinstance(attr, External):
            return attr.get()

        # return
        return attr

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
        valid = valid or isinstance(value, hylite.project.PMap)  # accept Pushbroom instances
        valid = valid or isinstance(value, External ) # accept external links
        valid = valid or isinstance(value, list) and np.array( [isinstance(d, hylite.HyData) ] for d in value ).all() # multimwl maps
        valid = valid or value is None # also accept None
        assert valid, "Error - %s is an invalid attribute type for HyCollection." % type(value)
        object.__setattr__(self, name, value)
"""
Implement decision tree function for classification. This is done in a generic form using decision_tree( ... ), but also
for specific minerals based on minimum wavelength data in subsequent functions.
"""

import numpy as np

def decision_tree( layers, labels ):
    """
    Implement a simple binary tree for assigning classes based on boolean input layers.

    *Arguments*:
     - layers = a list of boolean arrays indicating the value of each pixel/point at each successive layer in the tree.
     - labels = a dictionary keyed by leaves of the decision tree (tuples containing a boolean value for each level in
                the tree) with values that give class labels. Leaves that are not in the tree will be assigned
                a label of 0 (unknown). For example: { (True,False,True) : 1, (False,False,True) : 2 } will label the
                leaves of the tree True -> False -> True as 1 and False -> False -> True as 2, and all others as 0. Use
                None to skip branches for certain labels (e.g. True -> None -> False will ignore the second layer of
                the decision tree for this label).

    *Returns*:
     -  an output labels array of the sampe shape as each layer array but containing 0 ... n class labels.
    """

    out = np.zeros( layers[0].shape, dtype=np.int )
    names = ["Unknown"]
    for k in labels.items():
        key, value = k
        if not value in names:
            names.append(value)
        msk = np.full( layers[0].shape, True )
        for n,l in enumerate(layers):
            if n < len(key):
                if key[n] is not None:
                    msk = msk & (l == key[n])
        out[ msk ]=names.index(value)
    return out, names




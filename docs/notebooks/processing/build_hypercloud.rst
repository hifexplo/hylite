Build hypercloud
================

Load a directory of hyperspectral scenes that have defined camera
orientation data (see pose estimation notebook). If calibration data is
also defined (see identify calibration targets notebook) then an
atmospheric and topographic correction will also be performed.

The selected hypercloud derivatives are then calculated and exported.

-----------

WARNING: This notebook is now out of date. See documentation on hylite.hyscene and hylite.project.pmap for info
on projecting spectra from images onto point clouds to create a hypercloud.

-----------


.. code:: python

    import os
    import glob
    import numpy as np
    from tqdm.auto import tqdm
    import utm
    import re
    import matplotlib.pyplot as plt

.. code:: python

    import hylite
    import hylite.io as io
    from hylite import HyScene
    from hylite.correct import ELC, Panel
    from hylite.reference.spectra import Target
    from hylite.correct.topography import sph2cart

Define data directories
-----------------------

.. code:: python

    # path to point cloud hyperspectral data is being added to
    cloud_path = '/Users/thiele67/Documents/Data/CA/CA_50cm_coregistered.ply'
    export_directory = '/Users/thiele67/Documents/Data/CA/hypercloud'

.. code:: python

    # input directories (or images) to include in hypercloud
    path = [ 
        '/Users/thiele67/Documents/Data/CA/Terrestrial/2019',
        ]
    
    image_paths = []
    for p in path:
        if os.path.isdir(p):
            image_paths += glob.glob( os.path.join(p,"*.hdr"), recursive=True )
        else:
            image_paths.append(p)

.. code:: python

    print("Found %d images:" % len(image_paths))
    for i, p in enumerate(image_paths):
        print("%d - " % i, p)


.. parsed-literal::

    Found 8 images:
    0 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2019/Corta_Atalaya__85_0m00_2m00_0079.hdr
    1 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2019/Corta_Atalaya__86_0m00_2m00_0080.hdr
    2 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2019/Corta_Atalaya__88_0m00_2m00_0082.hdr
    3 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2019/Corta_Atalaya__84_0m00_2m00_0078.hdr
    4 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2019/Corta_Atalaya__87_0m00_2m00_0081.hdr
    5 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2020/CA_0080__1_1m00_2m00.hdr
    6 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2020/CA_0079__0_0m00_1m00.hdr
    7 -  /Users/thiele67/Documents/Data/CA/Terrestrial/2020/CA_0082__3_3m00_4m00.hdr


Define cloud export settings
----------------------------

.. code:: python

    #basic projection settings
    occ_tol = 0 #occlusion tolerance (in same units as cloud). 0 disables occlusions. 
    s = 3 # point size for projecting onto cloud. Must be an integer >= 1. 
    
    #topographic correction settings
    topo_correct = 'ambient' #'cfac' #'cfac' # topographic correction method to apply. Set to None to disable. 
                             # 'ambient', 'cfac' or 'minnaert' normally give best results. 
    high_thresh = 99 # pixels brighter than this percentile will be removed after the topo correction (removes false highlights)
    low_thresh = 0 # pixels darker than this percentile will be removed after the topo correction (uncorrected shadows)
    
    # atmospheric correction settings   
    atmos_correct = True # False # use target panels to apply atmospheric correction
    
    # colour correction settings
    colour_correct = True # perform colour correction
    reference_index = 0 # which image to use as reference for correction (match other images too)
    method = 'hist' # options are 'norm' (match mean and standard deviation) or 'hist' (match histograms)
    uniform = False # set to False to allow per-band colour correction (distorting the spectra).  
    
    #blending settings
    blend_mode = 'average' # options are "average",
                           #  "gsd" (use pixel with smallest footprint),
                           #  "weighted" (compute average weighted by gsd).
                
    # export settings
    export_hypercloud = False # create a hypercloud?
    vis = hylite.RGB # which bands should be mapped to hypercloud RGB
    export_bands = (0,-1) # put in band wavelengths to export (e.g. 2000.0, 2500.0), or (0,-1) to export all bands. 
    
    export_images = True # export corrected images? 

Run computer magics! ☀
----------------------

.. code:: python

    # create HyScenes
    scenes = []
    cloud = io.loadCloudPLY(cloud_path)
    for i,p in enumerate(image_paths):
        image = io.loadWithGDAL(p) # load image
        cam = image.header.get_camera()
        if cam is not None:
            print("Building scene %d... " % i, end='')
            scenes.append( HyScene(  image, cloud, cam, occ_tol = occ_tol, s=s ) )
        else:
            print("Failed. Image has no camera pose (%s)" % p)
    
    ##############################################################
    # apply topographic and atmospheric corrections
    ##############################################################
    uncorrected = [] # store scenes with no panel info here (we can calibrate them against corrected scenes [maybe])
    corrected = [] # scenes that have been successfully corrected
    for i,s in enumerate(scenes):
        
        print("Correcting scene %d..." % i, end='')
        
        # correct scene!
        suc = s.correct( atmos_correct, 
                         topo_correct is not None,
                         method = topo_correct, 
                         low_thresh = low_thresh,
                         high_thresh = high_thresh,
                         vb = True,
                         name = "Scene %d" % i, 
                         bands = vis,
                         topo_kwds = {})
        
        if suc: # success - move on to next one
            corrected.append(i)
            print(" Done.")
        else: # failed... why?
            if len(s.image.header.get_panel_names()) == 0: # no calibration panel
                uncorrected.append(i)
                print(" Missing panel.")
            elif not 'sun azimuth' in s.image.header: # no sun information for topo correction
                print(" Missing sun vector. Scene will not be corrected.")
                
    ##############################################################
    #Try to match scenes with no panel against corrected scenes
    ##############################################################
    max_points = 5000 # max number of pixels to calculate ELC for - make smaller to improve performance, 
                      # larger for better accuracy
        
    for i in uncorrected:
        
        print("Matching scene %d... " % i, end='')
    
        overlap = []
        overlap_size = []
        for n in corrected:
            px1, px2 = scenes[ i ].intersect_pixels( scenes[n] ) # get intersecting pixels
            overlap_size.append( len(px1) )
            overlap.append( (px1,px2) )
    
        best = np.argmax( overlap_size )
        if overlap_size[ best ] < 100:
            print(" insufficient overlap (%d pixels). Scene will not be corrected." % overlap_size[ best ])
            assert False
    
        print(" found %d overlapping pixels." % overlap_size[ best ])
    
        px1, px2 = overlap[best] # get overlapping pixels
        best = corrected[best] # convert to index in list of all scenes
    
        # subsample matches if too many points
        if px1.shape[0] > max_points:
            idx = np.random.choice( px1.shape[0], max_points, replace=False)
            px1 = px1[idx,:]
            px2 = px2[idx,:]
    
        # create suite of ELC objects assuming each pixel is a calibration target
        elc = []
        for p in tqdm(range(px1.shape[0])):
            rad = scenes[ i ].image.data[ px1[p,0], px1[p,1], : ]
            refl = scenes[ best ].image.data[ px2[p,0], px2[p,1], : ]
            t = Target( scenes[ best ].image.get_wavelengths(), refl, name="match" )
            elc.append( ELC( [ 
                        Panel( t, rad, wavelengths=scenes[ i ].image.get_wavelengths() )
                            ] ) ) 
    
        # average slope/intercept of elc 
        vals = []
        for e in elc:
            vals.append( [e.slope, e.intercept])
        vals = np.array(vals)
        m,c = np.nanmedian( vals, axis=0 )
    
        # create fake white reference and add to header
        refl = np.full( len(scenes[ i ].image.get_wavelengths()), 1.0 ) # pure white
        rad = (1 - c) / m
        t = Target( scenes[ i ].image.get_wavelengths(), refl, name="white_estimate")
        p = Panel( t, rad, wavelengths = scenes[ i ].image.get_wavelengths() )
        scenes[ i ].image.header.add_panel(p)
    
        # plot it for reference
        fig,ax = p.quick_plot()
        ax.set_title("Scenes %d: estimated (pure) white panel radiance" % i )
        fig.show()
        
        # apply correction
        print("Correcting scene %d..." % i, end='')
        
        # correct scene!
        s = scenes[i]
        suc = s.correct( atmos_correct, 
                         topo_correct is not None,
                         method = topo_correct, 
                         low_thresh = low_thresh,
                         high_thresh = high_thresh,
                         vb = True,
                         name = "Scene %d" % i, 
                         bands = vis,
                         topo_kwds = {})
        
        if suc: # success - move on to next one
            print(" Done.")
        else: # failed... why?
            if len(s.image.header.get_panel_names()) == 0: # no calibration panel
                print(" Missing panel.")
            elif not 'sun azimuth' in s.image.header: # no sun information for topo correction
                print(" Missing sun vector. Scene will not be corrected.")

.. code:: python

    ##############################################################
    # Apply colour corrections
    ##############################################################
    if colour_correct:
        for i,s in tqdm(enumerate(scenes), desc='Colour correction', total=len(scenes)):
            if i == reference_index: 
                continue # skip reference image
            s.match_colour_to( scenes[ reference_index ], method=method, uniform=uniform )
            
            # plot results
            fig,ax = s.quick_plot(hylite.RGB)
            ax.set_title("Colour corrected scene (RGB)")
            fig.show()
            fig,ax = s.image.plot_spectra(band_range=export_bands)
            ax.set_title("Colour corrected spectra")
            fig.show()

Export results
--------------

.. code:: python

    if export_images: # export corrected image
        for i, s in enumerate(scenes):
            name = os.path.splitext(os.path.basename( image_paths[i] ))[0] + '_refl.hdr'
            io.saveWithGDAL(os.path.join( export_directory, name ), s.image )
            
    if export_hypercloud: # build and export hypercloud
        hypercloud = HyScene.build_hypercloud( scenes, export_bands, blend_mode, trim=True, vb=True)
        hypercloud.colorise( vis, stretch=(1,99) )
        name = os.path.splitext(os.path.basename( image_paths[i] ))[0] + '_refl.hdr'
        hypercloud.compress()
        io.saveCloudPLY( os.path.join(export_directory, name), hypercloud)


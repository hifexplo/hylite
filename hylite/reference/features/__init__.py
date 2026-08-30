"""
This package contains reference features such as typical mineral absorptions or absorption positions of QAQC targets.
"""
from hylite.hyfeature import *

##############################
##Create basic/common features
##############################
#N.B. All values here are approximate and largely for (1) plotting and (2) as initial values for feature fitting
#     routines. Believe nothing!
class Features:
    """
    Specific absorption types. Useful for plotting etc. Not really used for anything and will probably be deprecated soon.
    """

    H2O = [ HyFeature("H2O", p, w, color='skyblue') for p,w in [(825,50), (940,75), (1130,100), (1400,150), (1900,200), (2700,150)] ]
    OH = [HyFeature("OH", 1400, 50, color='aquamarine'), HyFeature("OH", 1550, 50, color='aquamarine'), HyFeature("OH", 1800, 100, color='aquamarine')]
    AlOH = [HyFeature("AlOH", 2190, 60, color='salmon')]
    FeOH = [HyFeature("FeOH", 2265, 70, color='orange')]
    MgOH = [HyFeature("MgOH", 2330, 60, color='blue'), HyFeature("MgOH", 2385, 30,color='blue')]
    MgCO3 = [HyFeature("MgCO3", 2320, 20, color='green')]
    CaCO3 = [HyFeature("CaCO3", 2340, 20, color='blue')]
    FeCO3 = [HyFeature("FeCO3", 2350, 20, color='steelblue')]
    Ferrous = [HyFeature("Fe2+", 1000, 400, color='green')]
    Ferric = [HyFeature("Fe3+", 650, 170, color='green')]

    # REE features
    Pr = [HyFeature("Pr", w, 5, color=(74 / 256., 155 / 256., 122 / 256., 1)) for w in [457, 485, 473, 595, 1017] ]
    Nd = [HyFeature("Nd", w, 5, color=(116 / 256., 114 / 256., 174 / 256., 1)) for w in [430, 463, 475, 514, 525, 580, 627, 680, 750, 800, 880, 1430, 1720, 2335, 2470]]
    Sm = [HyFeature("Sm", w, 5, color=(116 / 256., 114 / 256., 174 / 256., 1)) for w in [945, 959, 1085, 1235, 1257, 1400, 1550]]
    Eu = [HyFeature("Eu", w, 5, color=(213 / 256., 64 / 256., 136 / 256., 1)) for w in [385, 405, 470, 530, 1900, 2025, 2170, 2400, 2610]]
    Dy = [HyFeature("Dy", w, 5, color=(117 / 256., 163 / 256., 58 / 256., 1)) for w in [368, 390, 403, 430, 452, 461, 475, 760, 810, 830, 915, 1117, 1276, 1725]]
    Ho = [HyFeature("Ho", w, 5, color=(222 / 256., 172 / 256., 59 / 256., 1)) for w in [363, 420, 458, 545, 650, 900, 1130, 1180, 1870, 1930, 2005]]
    Er = [HyFeature("Er", w, 5, color=(159 / 256., 119 / 256., 49 / 256., 1)) for w in [390, 405, 455, 490, 522, 540, 652, 805, 985, 1485, 1545]]
    Tm = [HyFeature("Tm", w, 5, color=(102 / 256., 102 / 256., 102 / 256., 1)) for w in [390, 470, 660, 685, 780, 1190, 1640, 1750]]
    Yb = [HyFeature("Yb", w, 5, color=(209 / 256., 53 / 256., 43 / 256., 1)) for w in [955, 975, 1004 ]]

# and some useful 'themes' (for plotting etc)
class Themes:
    """
    Some useful 'themes' (for plotting etc)
    """
    ATMOSPHERE = Features.H2O  #[HyFeature("H2O", 975, 30), HyFeature("H2O", 1395, 120), HyFeature("H2O", 1885, 180), HyFeature("H2O", 2450, 100)]
    OH = Features.AlOH + Features.FeOH + Features.MgOH
    DIAGNOSTIC = Features.Ferrous + Features.AlOH+Features.FeOH+Features.MgOH

#expose through HyFeature class for convenience
HyFeature.Features = Features
HyFeature.Themes = Themes
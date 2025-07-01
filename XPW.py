"""
XPW setup table
by Olivier Albert @ fastlite 

2 crystal XPW setup with the first crystal on focus
A telescope is used to focus on the first crystal:
it allows to obtain long focal length in a smaller footprint

Crystal spacing formula correspond to a fit of a graph summerazing all our experiments at various input energy, this graph was published here:
O. Albert, A. Jullien, J. Etchepare, S. Kourtev, N. Minkovski, and S. M. Saltiel, "Comment on "Generation of 1011 contrast 50 TW laser pulses"," Opt. Lett. 31, 2990-2992 (2006) 
The value provided should be considered as an empirical approximation, and not exact calculations.
But their strength is in the simplicity of the solution proposed: if you use those values it is going to work as expected (you get good XPW efficiency without major tuning)

This XPW dimensioning is from PCO group at Laboratoire d'Optique Appliquée, Palaiseau france

"""
# %%
import math
import numpy as np
print(' XPW setup parameters \n 2 crystals scheme and a telescope \n to focus on the first BaF2 with a long focal length on a small footprint \n')

# Pulse parameter
Diam = 2        # mm laser diameter entering XPW setup
Lambda = 1.03    # micron, central wavelength
pulse = 500 # fs, pulse duration
# telescope parameters
fa = 200.0      # mm
fb = -100.0      # mm
d = 130.0       # mm distance between fa and fb

def telescope_from_geometry(fa,fb,d):
    fab = (fa*fb)/(fa+fb-d)     # focale du telescope (mm)
    D = fab*(fa-d)/fa           # distance du miroir b au point focal (mm) 
    return fab, D

fab, D = telescope_from_geometry(fa,fb,d)
print(f' telescope:\n focal lengths ( {fa} mm, {fb} mm) and d= {d:.1f} mm\n equiv. focal length = {fab:.1f} mm at {D:.1f} mm from the last lens \n i.e. telescope footprint of {d+D:.1f} mm. \n')

def telescope_from_expected_focal_length(fa,fb,fab):
    d= fa+fb-(fa*fb)/fab        # distance between fa and fb
    D = fab*(fa-d)/fa           # distance du miroir b au point focal (mm) 
    return d , D
fab = np.round(fab/100)*100
d , D = telescope_from_expected_focal_length(fa,fb,fab)
print(f' telescope:\n focal lengths ( {fa} mm, {fb} mm) and d= {d:.1f} mm\n equiv. focal length = {fab:.1f} mm at {D:.1f} mm from the last lens \n i.e. telescope footprint of {d+D:.1f} mm. \n')


# %%

# BaF2 Sellmeier from : http://refractiveindex.info/?shelf=main&book=BaF2&page=Li
nbaf2 = math.sqrt(1.0+0.33973+0.81070/(1.0-math.pow(0.10065/Lambda,2))+0.19652/(1.0-math.pow(29.87/Lambda,2))+4.52469/(1.0-math.pow(53.82/Lambda,2)))

# Pulse parameters on focus (first BaF2)
waist = 2/math.pi*Lambda*fab/Diam   # Waist, micron
Div = Diam/(2*fab)                  # Divergence, rad

Dist = 8*Lambda/math.pi*math.pow(fab/Diam,2)*1.0E-3/3.6     # mm, distance between crystals

pcr = 3.77*math.pow(Lambda,2)/(8*math.pi*nbaf2*0.67E-13)    # Critical intensity for continuum generation (W/cm^2), calculated from Dharmadhikari 2006 OE
MaxE = pcr*math.pi*math.pow(1.2*waist,2)*pulse*1E-15*1E-5        # Pulse energy (mJ) to reach continuum on first XPW crystal with current setup

print(f' XPW parameters using telescope parameters from above (fab: {fab:.1f} mm): \n Working with a {pulse} fs pulse at {Lambda} um and an input beam diameter of {Diam} mm \n expected 30% XPW efficiency from max input of {MaxE:.3f} mJ (i.e. {MaxE*0.3:.3f} mJ) \n and a second BaF2 placed {Dist:.1f} mm from the first one.\n')

# %%
"""
Convenient little Python code that calculates the geometrical propagation of light rays in the thin lens approximation. 
Like most science oriented programs, this one requires numpy and matplotlib packages to work.
"""
import numpy as np, matplotlib.pyplot as plt
 
# ----------------------------------------------------------------
#        simply draws a thin-lens at the provided location
# parameters:
#     - z:    location along the optical axis (in mm)
#     - f:    focal length (in mm, can be negative if div. lens)
#     - diam: lens diameter in mm
#     - lbl:  label to identify the lens on the drawing
# ----------------------------------------------------------------
def add_lens(z, f, diam, lbl):
    ww, tw, rad = diam / 10.0, diam/3.0, diam / 2.0
    plt.plot([z, z],    [-rad, rad],                'k', linewidth=2)
    plt.plot([z, z+tw], [-rad, -rad+np.sign(f)*ww], 'k', linewidth=2)
    plt.plot([z, z-tw], [-rad, -rad+np.sign(f)*ww], 'k', linewidth=2)
    plt.plot([z, z+tw], [ rad,  rad-np.sign(f)*ww], 'k', linewidth=2)
    plt.plot([z, z-tw], [ rad,  rad-np.sign(f)*ww], 'k', linewidth=2)
    plt.plot([z+f, z+f], [-ww,ww], 'k', linewidth=2)
    plt.plot([z-f, z-f], [-ww,ww], 'k', linewidth=2)
    plt.text(z,rad+5.0, lbl, fontsize=12)
    plt.text(z,rad+2.0, 'f='+str(int(f)), fontsize=10)
 
# ----------------------------------------------------------------------
#      geometrical propagation of light rays from given source
# parameters:
#    - p0:  location of the source (z0, x0) along and off axis (in mm)
#    - NA:  numerical aperture of the beam (in degrees)
#    - nr:  number of rays to trace
#    - zl:  array with the location of the lenses
#    - ff:  array with the focal length of lenses
#    - lbl: label for the nature of the source
#    - col: color of the rays on plot
# ----------------------------------------------------------------------
def propagate_beam(p0, NA, nr, zl, ff, lbl='', col='b'):
    apa = NA*np.pi/180.0
    z0 = p0[0]
    if (np.size(p0) == 2): x0 = p0[1]
    else:                  x0 = 0.0
 
    zl1, ff1 = zl[(z0 < zl)], ff[(z0 < zl)]
    nl  = np.size(zl1) # number of lenses
 
    zz, xx, tani = np.zeros(nl+2), np.zeros(nl+2), np.zeros(nl+2)
    tan0 = np.tan(apa/2.0) - np.tan(apa) * np.arange(nr)/(nr-1)
 
    for i in range(nr):
        tani[0] = tan0[i] # initial incidence angle
        zz[0], xx[0] = z0, x0
        for j in range(nl):
            zz[j+1]   = zl1[j]
            xx[j+1]   = xx[j] + (zz[j+1]-zz[j]) * tani[j]
            tani[j+1] = tani[j] - xx[j+1] / ff1[j]
 
        zz[nl+1] = zmax
        xx[nl+1] = xx[nl] + (zz[nl+1]-zz[nl]) * tani[nl]
        plt.plot(zz, xx, col)
 
# ----------------------------------------------------------------------
#                            MAIN PROGRAM
# ----------------------------------------------------------------------
plt.clf()
 
zmin, zmax       = -100., 2*(d+D+Dist)+100.
xmin, xmax       = -25, 25
bignum, smallnum = 1e6, 1e-6   # all distances expressed in mm
 
# ------------------------------------
#   location + focal length of optics
# ------------------------------------
zl = np.array([0,d,d+D,d+D+Dist]) # lens positions
ff = np.array([fa,fb,bignum,bignum]) # lens focal length
 
xsrc, zsrc, zpup = 0.0 , 0.0, -bignum # position of src and pupil
srcpos = (zsrc, xsrc)
 
#  draw the different beams
# --------------------------
# propagate_beam(srcpos,          0.5, 50, zl, ff, 'src1', 'b')
# propagate_beam((0.0, -2.),      4,  20, zl, ff, 'src1', 'r')
propagate_beam((zpup,),    (Diam+1)/1E4,  50, zl, ff, 'src1', 'g')
# propagate_beam((110,),          2,  40, zl, ff, 'DM',   'y')

 
#  print a couple labels
# --------------------------
# plt.text(0, 20, 'src 1', bbox=dict(facecolor='blue', alpha=1), fontsize=10)
plt.text(0, -24, 'Olivier Albert', bbox=dict(facecolor='red',  alpha=1), fontsize=10)
# plt.text(0, 14, 'pupil', bbox=dict(facecolor='green',  alpha=1), fontsize=10)
# plt.text(0, 11, 'DM', bbox=dict(facecolor='yellow',  alpha=1), fontsize=10)
 
#      add the lenses
# -------------------------
for i in range(np.size(zl)): add_lens(zl[i], ff[i], 25, "L"+str(i))
 
#     plot optical axis
# -------------------------
plt.plot([zmin,zmax], [0,0], 'k')
plt.axis([zmin,zmax, xmin, xmax])
plt.title("XPW setup, 2crystals scheme \n telescope: L1 and L2, BaF2 crystals: L3 and L4")
plt.show()
# %%

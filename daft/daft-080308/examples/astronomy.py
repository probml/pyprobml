"""
Astronomical imaging
====================

This is a model for every pixel of every astronomical image ever
taken.  It is incomplete!

"""

from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([8, 6.75], origin=[0.5, 0.5], grid_unit=4., node_unit=1.4)

# Start with the plates.
tweak=0.02
rect_params = {"lw": 2}
pgm.add_plate(daft.Plate([1.5+tweak, 0.5+tweak, 6.0-2*tweak, 3.75-2*tweak], label=r"\Large telescope+camera+filter multiplets", rect_params=rect_params))
pgm.add_plate(daft.Plate([2.5+tweak, 1.0+tweak, 4.0-2*tweak, 2.75-2*tweak], label=r"\Large images", rect_params=rect_params))
pgm.add_plate(daft.Plate([3.5+tweak, 1.5+tweak, 2.0-2*tweak, 1.75-2*tweak], label=r"\Large pixel patches", rect_params=rect_params))
pgm.add_plate(daft.Plate([1.0+tweak, 4.25+tweak, 3.5-2*tweak, 1.75-2*tweak], label=r"\Large stars", rect_params=rect_params))
pgm.add_plate(daft.Plate([5.5+tweak, 4.25+tweak, 2.5-2*tweak, 1.75-2*tweak], label=r"\Large galaxies", rect_params=rect_params))

# ONLY pixels are observed
asp = 2.3
pgm.add_node(daft.Node("true pixels", r"~\\noise-free\\pixel patch", 5.0, 2.5, aspect=asp))
pgm.add_node(daft.Node("pixels", r"pixel patch", 4.0, 2.0, observed=True, aspect=asp))
pgm.add_edge("true pixels", "pixels")

# The sky
pgm.add_node(daft.Node("sky", r"sky model", 6.0, 2.5, aspect=asp))
pgm.add_edge("sky", "true pixels")
pgm.add_node(daft.Node("sky prior", r"sky priors", 8.0, 2.5, fixed=True))
pgm.add_edge("sky prior", "sky")

# Stars
pgm.add_node(daft.Node("star patch", r"star patch", 4.0, 3.0, aspect=asp))
pgm.add_edge("star patch", "true pixels")
pgm.add_node(daft.Node("star SED", r"~\\spectral energy\\distribution", 2.5, 4.75, aspect=asp+0.2))
pgm.add_edge("star SED", "star patch")
pgm.add_node(daft.Node("star position", r"position", 4.0, 4.75, aspect=asp))
pgm.add_edge("star position", "star patch")
pgm.add_node(daft.Node("temperature", r"temperature", 1.5, 5.25, aspect=asp))
pgm.add_edge("temperature", "star SED")
pgm.add_node(daft.Node("luminosity", r"luminosity", 2.5, 5.25, aspect=asp))
pgm.add_edge("luminosity", "star SED")
pgm.add_node(daft.Node("metallicity", r"metallicity", 1.5, 5.75, aspect=asp))
pgm.add_edge("metallicity", "star SED")
pgm.add_edge("metallicity", "temperature")
pgm.add_edge("metallicity", "luminosity")
pgm.add_node(daft.Node("mass", r"mass", 2.5, 5.75, aspect=asp))
pgm.add_edge("mass", "temperature")
pgm.add_edge("mass", "luminosity")
pgm.add_node(daft.Node("age", r"age", 3.5, 5.75, aspect=asp))
pgm.add_edge("age", "temperature")
pgm.add_edge("age", "luminosity")
pgm.add_node(daft.Node("star models", r"star models", 1.0, 4.0, fixed=True))
pgm.add_edge("star models", "temperature")
pgm.add_edge("star models", "luminosity")
pgm.add_edge("star models", "star SED")

# Galaxies
pgm.add_node(daft.Node("galaxy patch", r"galaxy patch", 5.0, 3.0, aspect=asp))
pgm.add_edge("galaxy patch", "true pixels")
pgm.add_node(daft.Node("galaxy SED", r"~\\spectral energy\\distribution", 6.5, 4.75, aspect=asp+0.2))
pgm.add_edge("galaxy SED", "galaxy patch")
pgm.add_node(daft.Node("morphology", r"morphology", 7.5, 4.75, aspect=asp))
pgm.add_edge("morphology", "galaxy patch")
pgm.add_node(daft.Node("SFH", r"~\\star-formation\\history", 7.5, 5.25, aspect=asp))
pgm.add_edge("SFH", "galaxy SED")
pgm.add_edge("SFH", "morphology")
pgm.add_node(daft.Node("galaxy position", r"~\\redshift\\ \& position", 6.0, 5.25, aspect=asp))
pgm.add_edge("galaxy position", "galaxy SED")
pgm.add_edge("galaxy position", "morphology")
pgm.add_edge("galaxy position", "galaxy patch")
pgm.add_node(daft.Node("dynamics", r"orbit structure", 6.5, 5.75, aspect=asp))
pgm.add_edge("dynamics", "morphology")
pgm.add_edge("dynamics", "SFH")
pgm.add_node(daft.Node("galaxy mass", r"mass", 7.5, 5.75, aspect=asp))
pgm.add_edge("galaxy mass", "dynamics")
pgm.add_edge("galaxy mass", "galaxy SED")
pgm.add_edge("galaxy mass", "SFH")

# Universals
pgm.add_node(daft.Node("extinction model", r"~\\extinction\\model", 5.0, 4.75, aspect=asp))
pgm.add_edge("extinction model", "star patch")
pgm.add_edge("extinction model", "galaxy patch")
pgm.add_node(daft.Node("MW", r"~\\Milky Way\\formation", 4.0, 6.5, aspect=asp))
pgm.add_edge("MW", "metallicity")
pgm.add_edge("MW", "mass")
pgm.add_edge("MW", "age")
pgm.add_edge("MW", "star position")
pgm.add_edge("MW", "extinction model")
pgm.add_node(daft.Node("galaxy formation", r"~\\galaxy\\formation", 5.0, 6.5, aspect=asp))
pgm.add_edge("galaxy formation", "MW")
pgm.add_edge("galaxy formation", "dynamics")
pgm.add_edge("galaxy formation", "galaxy mass")
pgm.add_edge("galaxy formation", "extinction model")
pgm.add_node(daft.Node("LSS", r"~\\large-scale\\structure", 6.0, 6.5, aspect=asp))
pgm.add_edge("LSS", "galaxy position")
pgm.add_node(daft.Node("cosmology", r"~\\cosmological\\parameters", 6.0, 7.0, aspect=asp))
pgm.add_edge("cosmology", "LSS")
pgm.add_edge("cosmology", "galaxy formation")
pgm.add_node(daft.Node("god", r"God", 7.0, 7.0, fixed=True))
pgm.add_edge("god", "cosmology")

# Sensitivity
pgm.add_node(daft.Node("zeropoint", r"~\\zeropoint\\(photocal)", 3.0, 3.0, aspect=asp))
pgm.add_edge("zeropoint", "true pixels")
pgm.add_node(daft.Node("exposure time", r"exposure time", 3.0, 2.5, observed=True, aspect=asp))
pgm.add_edge("exposure time", "zeropoint")

# The PSF
pgm.add_node(daft.Node("WCS", r"~\\astrometric\\calibration", 3.0, 2.0, aspect=asp))
pgm.add_edge("WCS", "star patch")
pgm.add_edge("WCS", "galaxy patch")
pgm.add_node(daft.Node("psf", r"PSF model", 3.0, 3.5, aspect=asp))
pgm.add_edge("psf", "star patch")
pgm.add_edge("psf", "galaxy patch")
pgm.add_node(daft.Node("optics", r"optics", 2.0, 3.0, aspect=asp-1.2))
pgm.add_edge("optics", "psf")
pgm.add_edge("optics", "WCS")
pgm.add_node(daft.Node("atmosphere", r"~\\atmosphere\\model", 1.0, 3.5, aspect=asp))
pgm.add_edge("atmosphere", "psf")
pgm.add_edge("atmosphere", "WCS")
pgm.add_edge("atmosphere", "zeropoint")

# The device
pgm.add_node(daft.Node("flatfield", r"flat-field", 2.0, 1.5, aspect=asp))
pgm.add_edge("flatfield", "pixels")
pgm.add_node(daft.Node("nonlinearity", r"non-linearity", 2.0, 1.0, aspect=asp))
pgm.add_edge("nonlinearity", "pixels")
pgm.add_node(daft.Node("pointing", r"~\\telescope\\pointing etc.", 2.0, 2.0, aspect=asp))
pgm.add_edge("pointing", "WCS")
pgm.add_node(daft.Node("detector", r"detector priors", 1.0, 1.5, fixed=True))
pgm.add_edge("detector", "flatfield")
pgm.add_edge("detector", "nonlinearity")
pgm.add_node(daft.Node("hardware", r"hardware priors", 1.0, 2.5, fixed=True))
pgm.add_edge("hardware", "pointing")
pgm.add_edge("hardware", "exposure time")
pgm.add_edge("hardware", "optics")

# Noise
pgm.add_node(daft.Node("noise patch", r"noise patch", 5.0, 2.0, aspect=asp))
pgm.add_edge("noise patch", "pixels")
pgm.add_edge("true pixels", "noise patch")
pgm.add_node(daft.Node("noise model", r"noise model", 7.0, 2.0, aspect=asp))
pgm.add_edge("noise model", "noise patch")
pgm.add_node(daft.Node("noise prior", r"noise priors", 8.0, 2.0, fixed=True))
pgm.add_edge("noise prior", "noise model")
pgm.add_node(daft.Node("cosmic rays", r"~\\cosmic-ray\\model", 8.0, 1.5, aspect=asp))
pgm.add_edge("cosmic rays", "noise patch")

# Render and save.
pgm.render()
pgm.figure.savefig("astronomy.pdf")
pgm.figure.savefig("astronomy.png", dpi=150)

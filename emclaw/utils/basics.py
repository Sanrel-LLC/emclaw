import numpy as np
import pickle
import os

# Function to calculate grid parameters
def grid_basic(dim_lims, cfl, co=1.0, v=1.0):
    """
    Calculate grid parameters.

    grid_basic(x_lower, x_upper, mx, cfl, co = 1, v = 1)


    Args:
        x_lower (dbl): value of x_lower (grid)
        x_upper (dbl): value of x_upper (grid)
        mx (int): number of cells
        cfl (float): Desired CFL number
        co (float, optional): Speed of light, from material.co. Defaults to 1.0.
        v (float, optional): Speed of wave, from source.v. Defaults to 1.0.
        dim_lims (list of tuples): List of tuples containing (x_lower, x_upper, mx)

    Returns:
        tuple: Tuple containing (dx1, dx2, ..., dxi, dt, tf)
    """
    dj = [(ds[1] - ds[0]) / ds[2] for ds in dim_lims]
    dt_div = np.sum([1.0 / x ** 2 for x in dj])
    dt = 0.9 * cfl / (co * np.sqrt(dt_div))
    tf = 2.0 * (dim_lims[0][1] - dim_lims[0][0]) / np.min(v)

    return *dj, dt, tf 

# Function to set output directories for material and source
def set_outdirs(material, source, outdir, debug):
    """
    Set output directories for material and source.

    set_outdirs(material, source, outdir, debug)

    Args:
        material (instance): Instance of material class
        source (instance): Instance of source class
        outdir (string): Directory where to save the data
        debug (bool): Whether to do _dump_to_latex for material and source
    """
    material._outdir = outdir
    source._outdir = outdir
    if debug:
        material._dump_to_latex()
        source._dump_to_latex()
    return

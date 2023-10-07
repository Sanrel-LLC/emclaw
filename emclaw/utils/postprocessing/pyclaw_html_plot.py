# Import necessary libraries and modules
import os
import sys
import shutil
import errno
from glob import glob
from clawpack.petclaw import plot
import matplotlib

# Set the backend for non-interactive plotting
matplotlib.use('Agg')

# Update matplotlib's font size
matplotlib.rcParams.update({'font.size': 10})

# Define a function to copy directories
def copy(src, dest):
    try:
        # Try to copy the source directory to the destination
        shutil.copytree(src, dest)
    except OSError as e:
        # Handle exceptions, such as if the source isn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            print('Directory not copied. Error: %s' % e)

# Define a function to generate HTML plots from a directory
def html_plot(src):
    plot.html_plot(outdir=src)

# Define the main plotting function
def main_plot(outdir='./_output', multiple=False, overwrite=False):
    if multiple:
        outdir = outdir + '*'
    
    # Get a list of directories that match the pattern
    outdirs = sorted(glob(outdir))
    
    for dirs in outdirs:
        if overwrite or not os.path.exists(os.path.join(outdir, '_plots')):
            # Generate HTML plots for the directory
            plot.html_plot(outdir=dirs)
            # Copy the '_plots' directory to the output directory
            copy('./_plots', os.path.join(dirs, '_plots'))     
            # Remove the temporary '_plots' directory
            shutil.rmtree('./_plots')

# Main execution block
if __name__ == "__main__":
    from clawpack.pyclaw import util
    args, app_args = util._info_from_argv(sys.argv)
    # Call the main_plot function with command-line arguments
    main_plot(**app_args)
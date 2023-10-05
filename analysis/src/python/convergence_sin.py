import os
from convergence import Errors1D

# Define the path to the directory containing MATLAB results
matpath = '../matlab/results'

# Define the MATLAB source file name
matsrc = '_sin_nc_65536.mat'

# Define the directory where 1D test results are stored
testdir = '/simdesk/sandbox/emclaw/results/1D/_convergence_sin_homogeneous_tfluct'

# Define the directory for comparison with another result
compare_dir = '/simdesk/sandbox/emclaw/results/1D/_convergence_sin'

# Define the base directory name prefix
basedir = '_output_'

# Define the minimum and maximum values for 'basedir'
basemin = 7
basemax = 15

# Define the frame number
frame = 5

# Define the directory where the summary will be saved
savedir = os.path.join(testdir, '_summaryb')

# Create an instance of the Errors1D class
error = Errors1D(testdir, basedir, savedir, frame)

# Set the 'matsrc' attribute to the path of the MATLAB source file
error.matsrc = os.path.join(matpath, matsrc)

# Set the 'finesrc' attribute to the directory containing 'basedir16'
error.finesrc = os.path.join(testdir, basedir + '16')

# Set the 'basemin' and 'basemax' attributes
error.basemin = basemin
error.basemax = basemax

# Enable debugging mode
error.debug = True

# Define the range of P-line for analysis
error.p_line_range = [1, 7]

# Set the homogeneous flag to True
error.homogeneous = True

# Set the directory for comparison
error.compare_dir = compare_dir

# Enable comparison mode
error.compare = True

# Call the 'convergence' method to perform convergence analysis
error.convergence()

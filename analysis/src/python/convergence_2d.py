  import os
from convergence import Errors2D

# Define the directory where test results are stored
testdir = '/media/shaheen/dev/results/maxwell_2d/convergence_averaged'

# Define the base directory name prefix
basedir = '_output_plane_'

# Define the minimum and maximum values for 'basedir'
basemin = 7
basemax = 13

# Define the frame number
frame = 45

# Define the directory where the summary will be saved
savedir = '/simdesk/sandbox/emclaw/results/2D/convergence_averaged_test/summary'

# Create an instance of the Errors2D class
error = Errors2D(testdir, basedir, savedir, frame)

# Set the 'finesrc' attribute to the directory containing the highest 'basedir' value ('basedir13')
error.finesrc = os.path.join(testdir, basedir + '13')

# Set the 'basemin' and 'basemax' attributes
error.basemin = basemin
error.basemax = basemax

# Enable debugging mode
error.debug = True

# Call the 'convergence' method to perform convergence analysis
error.convergence()

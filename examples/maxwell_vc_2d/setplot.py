# Import necessary modules and libraries
import numpy as np

# Define a function to set up plot figures, axes, and items
def setplot(plotdata):
    """
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """
    
    import matplotlib.cm as colormaps

    plotdata.clearfigures()  # clear any old figures, axes, items data

    # Figure for pcolor plot (Hz)
    plotfigure = plotdata.new_plotfigure(name='Hz', figno=0)
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = '$H_z$'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 2
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Figure for pcolor plot (Ey)
    plotfigure = plotdata.new_plotfigure(name='Ey', figno=1)
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = '$E_y$'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 1
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Figure for pcolor plot (Ex)
    plotfigure = plotdata.new_plotfigure(name='Ex', figno=2)
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = '$E_x$'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = 0
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Figure for pcolor plot (S)
    plotfigure = plotdata.new_plotfigure(name='S', figno=3)
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = '$S$'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = poynting
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Figure for total field intensity and refractive index (line across mid x)
    plotfigure = plotdata.new_plotfigure(name='I,n', figno=4)
    
    # Plot fields intensity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = ''

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = intensity_line
    plotitem.plotstyle = ':'
    plotitem.color = 'b'
    plotitem.show = True
    plotitem.kwargs = {'linewidth': 1, 'markersize': 1}

    # Plot refractive index
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = ''

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = refind_line
    plotitem.plotstyle = ':'
    plotitem.color = 'g'
    plotitem.show = True
    plotitem.kwargs = {'linewidth': 1, 'markersize': 1}

    # Figure for total field energy and refractive index (2D)
    plotfigure = plotdata.new_plotfigure(name='S,n', figno=5)

    # Plot fields intensity
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = ''

    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = poynting
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Plot refractive index
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = ''

    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = refind
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Figure for pcolor plot (n)
    plotfigure = plotdata.new_plotfigure(name='n', figno=6)
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = '$n$'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = refind
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Figure for pcolor plot (dn)
    plotfigure = plotdata.new_plotfigure(name='dn', figno=7)
    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = 'auto'
    plotaxes.title = '$n$'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='2d_pcolor')
    plotitem.plot_var = dn
    plotitem.pcolor_cmap = colormaps.jet
    plotitem.add_colorbar = True
    plotitem.show = True

    # Parameters used only when creating HTML and/or LaTeX hardcopy
    # e.g., via visclaw.frametools.printframes:
    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create HTML files of plots?
    plotdata.html_homelink = '../README.html'   # pointer for top of index
    plotdata.latex = True                    # create LaTeX file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    return plotdata

# Define functions for calculating variables
def refind(current_data):
    n = np.sqrt(current_data.aux[1,:,:] * current_data.aux[2,:,:])
    return n

def dn(current_data):
    dn = current_data.aux[4,:,:]
    return dn

def efield(current_data):
    ef = current_data.q[1,:,:]
    return ef

def hfield(current_data):
    hf = current_data.q[2,:,:]
    return hf

def intensity(current_data):
    I = np.sqrt(current_data.q[1,:,:]**2 + current_data.q[2,:,:]**2)
    return I

def intensity_line(current_data):
    p = current_data.q[0,:,:].shape[1] / 2
    I = np.sqrt(current_data.q[1,:,p]**2 + current_data.q[2,:,p]**2)
    return I

def refind_line(current_data):
    p = current_data.q[0,:,:].shape[1] / 2
    n = np.sqrt(current_data.aux[1,:,p]**2 + current_data.aux[2,:,p]**2)
    return n

def poynting(current_data):
    sx = current_data.q[1,:,:] * current_data.q[2,:,:]
    sy = -1.0 * current_data.q[0,:,:] * current_data.q[2,:,:]
    s = np.sqrt(sx**2 + sy**2)
    return s
import numpy as np


heading = {}
heading['xyz'] = [0,1,2]
heading['xy' ] = [0,1]
heading['xz' ] = [0,2]
heading['yz' ] = [1,2]
heading['x'  ] = [0]
heading['y'  ] = [1]
heading['z'  ] = [2]

# polarization is complementary to heading
polarization = {}
polarization['xyz'] = [0,1,2]
polarization['xy' ] = [1,0]
polarization['xz' ] = [2,0]
polarization['yz' ] = [2,1]
polarization['x'  ] = [1]
polarization['y'  ] = [0]
polarization['z'  ] = [0]

def user_source():
    pass

def user_transversal_source():
    pass

class Sources:
    def _unpack_options(self,options={}):
        # first add the options passed to the function
        for key in options:
            setattr(self,key,options[key])

        # unpack options from self.options{}
        for key in self.options:
            setattr(self,key,self.options[key])

   def _set_f_w(self, material, dictin):
    # Set the 'c' attribute of the object to the speed of light of the given material.
    setattr(self, 'c', material.co)

    # Iterate through the key-value pairs in the 'dictin' dictionary.
    for key, value in dictin.items():
        # Set attributes of the object with the keys and values from the dictionary.
        setattr(self, key, value)

    # Check if either 'wavelength' or 'omega' is defined, calculate the missing one if needed.
    if hasattr(self, 'wavelength'):
        setattr(self, 'omega', 2.0 * np.pi * material.co / self.wavelength)
    elif hasattr(self, 'omega'):
        setattr(self, 'wavelength', 2.0 * np.pi * material.co / self.omega)
    else:
        msg = 'You must define either wavelength or omega'
        print(msg)

def gaussian(self, x, dx, s=1.0, xo=0.0, v=0.0, t=0.0):
    try:
        from scipy.special import erf
    except:
        self.averaged = False

    arg = xo + v * t - x

    if self.averaged:
        print('averaging')
        ddx = dx / 2.0
        # Calculate the Gaussian error function with averaging.
        erravg = (np.sqrt(np.pi) * s * (erf((ddx + arg) / s) + erf((ddx - arg) / s))) / (2.0 * dx)
    else:
        # Calculate the Gaussian without averaging.
        erravg = np.exp(-arg**2 / s**2)

    if self.cut:
        print('cutting')
        # Apply a cutoff condition if 'cut' attribute is True.
        span = ((arg**2 / s**2) <= (4.0 * s))
        erravg = erravg * span

    return erravg

def harmonic(self, x, dx, xo=0.0, omega=0.0, k=1.0, t=0.0, f=None):
    arg = x - xo - (omega / k) * t
    if f is None:
        f = self.harmonic_function

    if self.averaged:
        ddx = dx / 2.0
        if f.__name__ == 'sin':
            # Calculate harmonic with sine function for averaging.
            erravg = (np.cos(k * (ddx - arg)) - np.cos(k * (ddx + arg))) / (k * dx)
        elif f.__name__ == 'cos':
            # Calculate harmonic with cosine function for averaging.
            erravg = (np.sin(k * (ddx - arg)) + np.sin(k * (ddx + arg))) / (k * dx)
    else:
        # Calculate harmonic using the specified function 'f'.
        erravg = f(k * arg)

    return erravg

def init(self, state):
    # Initialize the 'q' component of the state with zeros.
    state.q.fill(0.0)

    if self.shape == 'off':
        dimh = heading[self.heading]
        dimp = polarization[self.heading]
        grid = state.grid

        if state.num_dim == 1:
            x = grid.x.centers
            # Generate a Gaussian waveform and set it in the appropriate 'q' component.
            waveshape = self.gaussian(x, self._dx, xo=self.offset, s=self.pulse_width)
            state.q[0, :] = self._material.zo * waveshape
            state.q[1, :] = waveshape

        if state.num_dim >= 2:
            waveshape = 1.0
            for i in range(len(self.heading)):
                h = dimh[i]
                # Generate a multi-dimensional Gaussian waveform.
                waveshape = waveshape * self.gaussian(grid.c_centers[h], self._delta[h], xo=self.offset[h], s=self.pulse_width[h])

            if len(self.heading) == 1:
                if state.num_dim == 2:
                    waveshape = self.transversal_function(grid.c_centers[dimp[0]]) * waveshape
                state.q[dimp[0], :, :] = ((-1.0) ** dimh[0]) * self._material.zo * waveshape

            if state.num_dim == 2:
                p = 2
            if state.num_dim == 3:
                p = 5

            state.q[p, :, :] = waveshape

    return state


   def scattering_bc(self, state, dim, t, qbc, auxbc, num_ghost):
    # Compute cell centers with ghost cells for the grid
    grid = state.grid
    grid.compute_c_centers_with_ghost(num_ghost, recompute=True)
    t = state.t  # Set 't' to the current time from the state

    if state.num_dim == 1:
        x = grid.x.centers_with_ghost[:num_ghost]
        qbc[:, :num_ghost] = self.function(x, t)  # Apply the 'function' to 'x' and store in 'qbc'

    if state.num_dim == 2:
        # Separate cell centers into 'x' and 'y'
        x = grid._c_centers_with_ghost[0]
        y = grid._c_centers_with_ghost[1]
        if dim.name == state.grid.dimensions[0].name:
            x = x[:num_ghost, :]
            y = y[:num_ghost, :]
            qbc[:, :num_ghost, :] = self.function(x, y, t)
        else:
            x = x[:, :num_ghost]
            y = y[:, :num_ghost]
            qbc[:, :, :num_ghost] = self.function(x, y, t)

    if state.num_dim == 3:
        # Separate cell centers into 'x', 'y', and 'z'
        x = grid._c_centers_with_ghost[0]
        y = grid._c_centers_with_ghost[1]
        z = grid._c_centers_with_ghost[2]
        if dim.name == state.grid.dimensions[0].name:
            x = x[:num_ghost, :, :]
            y = y[:num_ghost, :, :]
            z = z[:num_ghost, :, :]
            qbc[:, :num_ghost, :, :] = self.function(x, y, z, t)
        if dim.name == state.grid.dimensions[1].name:
            x = x[:, :num_ghost, :]
            y = y[:, :num_ghost, :]
            z = z[:, :num_ghost, :]
            qbc[:, :, :num_ghost, :] = self.function(x, y, z, t)
        if dim.name == state.grid.dimensions[2].name:
            x = x[:, :, :num_ghost]
            y = y[:, :, :num_ghost]
            z = z[:, :, :num_ghost]
            qbc[:, :, :, :num_ghost] = self.function(x, y, z, t)

    return qbc

def dump(self):
    # Print attribute names and their values for the current object
    for attr in sorted(dir(self)):
        if not attr.startswith('_'):
            print("%s = %s" % (attr, getattr(self, attr)))

def _dump_to_latex(self):
    # Convert object attributes into LaTeX table format and write to a file
    from tabulate import tabulate

    # Define the LaTeX table header
    strt = r'\begin{table][h!]' + '\n' + r'\centering' + '\n' + r'\begin{tabular}[cl]' + '\n' + r'\hline' + '\n'
    strt = strt + r'variable & value(s) \\' + '\n' + r'\hline' + '\n'

    for attr in sorted(dir(self)):
        if not attr.startswith('_'):
            s = getattr(self, attr)
            if isinstance(s, str):
                # If the attribute is a string, format it in LaTeX
                strt = strt + '\t' + r'\verb+' + attr + '+ \t' + r'&' + '\t' + s + r' \\' + '\n'
            elif isinstance(s, float):
                # If the attribute is a float, format it in LaTeX
                strt = strt + '\t' + r'\verb+' + attr + '+ \t' + r'&' + '\t' + str(s) + r' \\' + '\n'
            elif isinstance(s, bool):
                # If the attribute is a boolean, format it in LaTeX
                strt = strt + '\t' + r'\verb+' + attr + '+ \t' + r'&' + '\t' + str(s) + r' \\' + '\n'
            else:
                try:
                    len(s)
                    # If the attribute is iterable (e.g., a list or array), format it in LaTeX
                    strt = strt + '\t' + r'\multicolumn{1}{c}\multirow{' + str(np.shape(s)[0]) + r'}{*}{\verb+' + attr + r'+}' + '\t' + r'&' + '\t'
                    for k in range(np.shape(s)[0]):
                        strt = strt + str(s[k]) + r' \\'
                    strt = strt + '\n'
                except:
                    # If the attribute doesn't fit the above cases, format it in LaTeX
                    if ('function' in str(s)): s = str(s).split('function ')[1].split('at')[0]
                    if ('method' in str(s)): s = str(s).split('method')[1].split('at')[0].split('.')[1].split('of')[0]
                    if ('ufunc' in str(s)): s = str(s).split('ufunc ')[1].split('>')[0]
                    strt = strt + '\t' + r'\verb+' + attr + '+ \t' + r'&' + '\t' + str(s) + r' \\' + '\n'

    # Complete the LaTeX table and write it to a file
    strt = strt + r'\end{tabular}' + '\n' + r'\end{table]' + '\n'
    import uuid
    import os
    try:
        os.makedirs(self._outdir)
    except:
        pass
    f = open(os.path.join(self._outdir, '_source_' + str(uuid.uuid1()) + '.tex'), 'a')
    f.write(strt)
    f.close()

def __init__(self):
    # Initialize object attributes with default values
    self.shape = None
    self.custom = False
    self.custom_func = user_source
    self.heading = 'x'
    self._outdir = './'

class Source1D(Sources):

    def setup(self, options={}):
        # Set up the source object based on provided options
        self._unpack_options(options=options)
        self.pulse_width = self.wavelength

        if self.shape == 'plane':
            # Define the function to use for a 'plane' source shape
            self.harmonic_function = np.sin
            self.function = self._plane

        if self.shape == 'pulse':
            # Define the function to use for a 'pulse' source shape
            self.shape_function = np.exp
            self.function = self._pulse
            self.averaged = True
            self._dx = 1.0
            self._cp = np.sqrt(np.pi)

        if self.shape == 'harmonic pulse':
            # Define the function to use for a 'harmonic pulse' source shape
            self.harmonic_function = np.sin
            self.shape_function = np.exp
            self.function = self._harmonic_pulse

        if self.shape == 'off':
            # Define the function to use for a source shape 'off'
            self.shape_function = np.exp
        return


  import numpy as np

class Sources:
    def _plane(self, x, t):
        # Initialize a 2xlen(x) numpy array with zeros.
        wave = np.zeros([2, len(x)], order='F')
        
        # Calculate the harmonic wave.
        harmonic = self.harmonic(x, self._dx, omega=self.omega, k=self.k, t=t)

        # Set the electric field component.
        wave[0, :] = self.Ey * harmonic

        # Set the magnetic field component.
        wave[1, :] = self.Hz * harmonic

        return wave

    def _pulse(self, x, t):
        # Initialize a 2xlen(x) numpy array with zeros.
        wave = np.zeros([2, len(x)], order='F')
        
        # Calculate the pulse shape.
        pulseshape = self.gaussian(x, self._dx, xo=self.offset, v=self.v, t=t, s=self.pulse_width)

        # Set the electric field component.
        wave[0, :] = self.Ey * pulseshape

        # Set the magnetic field component.
        wave[1, :] = self.Hz * pulseshape

        return wave

    def _harmonic_pulse(self, x, t):
        # Initialize a 2xlen(x) numpy array with zeros.
        wave = np.zeros([2, len(x)], order='F')

        # Calculate the harmonic wave.
        harmonic = self.harmonic(x, self._dx, omega=self.omega, k=self.k, t=t)

        # Calculate the pulse shape.
        pulseshape = self.gaussian(x, self._dx, xo=self.offset, v=self.v, t=t, s=self.pulse_width)

        # Set the electric field component.
        wave[0, :] = self.Ey * harmonic * pulseshape

        # Set the magnetic field component.
        wave[1, :] = self.Hz * harmonic * pulseshape

        return wave

    def _off(self, x, t=0):
        # Initialize a 2xlen(x) numpy array with zeros.
        wave = np.zeros([2, len(x)], order='F')

        return wave

    def __init__(self, material, shape='plane', **kwargs):
        # Initialize the source with material and options.
        self._set_f_w(material, kwargs)
        self.options = {}
        self.k = 2.0 * np.pi / self.wavelength
        self.v = material.co / material.bkg_n
        self.Ey = material.zo
        self.Hz = 1.0
        self.offset = 0.0
        self.shape = shape
        self.custom = False
        self.function = None
        self.averaged = True
        self.custom_func = user_source
        self._material = material
        self.dx = 1.0
        self.heading = 'x'
        self.cut = False
        self.num_dim = 1

class Source2D(Sources):

    def setup(self, options={}):
        # Unpack options.
        self._unpack_options(options=options)
        self.pulse_width = self.wavelength * np.ones([2])

        if self.shape == 'custom':
            self.custom = True

        if self.custom:
            self.shape = 'custom'
            self.custom_function = user_source

        if self.shape == 'plane':
            self.harmonic_function = np.sin
            self.function = self._plane

        if self.shape == 'pulse':
            self.shape_function = np.exp
            self.function = self._pulse
            self.t_off = (4.0 * self.pulse_width[0]) / self.v[0]

        if self.shape == 'harmonic pulse':
            self.harmonic_function = np.sin
            self.shape_function = np.exp
            self.function = self._harmonic_pulse

        if self.shape == 'bessel pulse':
            self.bessel_order = 0
            self.function = self._bessel_pulse
            self.kill_after_first_zero = True

        if self.shape == 'off':
            self.shape_function = np.exp

        if self.transversal_shape == 'plane':
            self.transversal_function = lambda y: 1.0

        if self.transversal_shape == 'gauss':
            self.transversal_function = lambda y: self._transversal_gauss(y)

        if self.transversal_shape == 'cosine':
            self.transversal_function = lambda y: self._transversal_cosine(y)

        if self.transversal_shape == 'bessel':
            self.transversal_bessel_order = 0
            self.transversal_kill_after_first_zero = True
            self.transversal_function = self._transversal_bessel

        return

   # Import necessary libraries
import numpy as np
from scipy.special import jn, jn_zeros

# Define a class called 'Sources'
class Sources:
    # Define a function to set transversal plane shape
    def _trasversal_plane(self, u):
        shape = 1.0
        return shape

    # Define a function to calculate transversal cosine shape
    def _transversal_cosine(self, u):
        p = polarization[self.heading][0]
        du = self._delta[p]
        uo = self.transversal_offset
        shape = self.harmonic(u, du, xo=uo, k=np.pi / self.transversal_width, f=np.cos)
        r = (u - uo) / self.transversal_width
        shape = shape * (np.abs(r) <= 0.5)
        return shape

    # Define a function to calculate transversal Gaussian shape
    def _transversal_gauss(self, u):
        p = polarization[self.heading]
        du = self._delta[p]
        shape = self.gaussian(u, du, xo=self.transversal_offset, s=self.transversal_width)
        return shape

    # Define a function to calculate transversal Bessel shape
    def _transversal_bessel(self, y):
        first_zero = jn_zeros(self.transversal_bessel_order, 1)
        shape = jn(self.transversal_bessel_order, (y - self.transversal_offset) * (first_zero[0]) / (self.transversal_width / 2.0))
        if self.transversal_kill_after_first_zero:
            shape_kill = np.abs((y - self.transversal_offset) * (first_zero[0]) / (self.transversal_width / 2.0)) <= (first_zero[0])
            shape = shape_kill * shape
        return shape

    # Define a function to calculate the plane wave
    def _plane(self, x, y, t=0):
        wave = np.zeros([3, x.shape[0], y.shape[1]], order='F')
        harmonic = self.transversal_function(y) * self.harmonic(x, self._delta[0], k=self.k[0], omega=self.omega, t=t)
        wave[0, :, :] = self.amplitude[0] * harmonic
        wave[1, :, :] = self.amplitude[1] * harmonic
        wave[2, :, :] = self.amplitude[2] * harmonic
        return wave

    # Define a function to calculate the pulse shape
    def _pulse(self, x, y, t=0):
        wave = np.zeros([3, x.shape[0], y.shape[1]], order='F')
        dimh = heading[self.heading]
        dimp = polarization[self.heading]
        if t <= self.t_off:
            shape = 1.0
            shape = shape * self.gaussian(x, self._delta[0], xo=self.offset[0], s=self.pulse_width[0], v=self.v[0], t=t)
            shape = self.transversal_function(y) * shape
        else:
            shape = 0.0
        if len(self.heading) == 1:
            wave[dimp[0], :, :] = ((-1.0) ** dimh[0]) * self._material.zo * shape
        wave[2, :, :] = self.amplitude[2] * shape
        return wave

    # Define a function to calculate the harmonic pulse shape
    def _harmonic_pulse(self, x, y, t=0):
        wave = np.zeros([3, x.shape[0], y.shape[1]], order='F')
        if t <= self.t_off:
            harmonic = self._plane(x, y, t)
            shape = self._pulse(x, y, t)
            shape = shape * harmonic
        else:
            shape = 0.0
        wave[0, :, :] = self.amplitude[0] * shape[0]
        wave[1, :, :] = self.amplitude[1] * shape[1]
        wave[2, :, :] = self.amplitude[2] * shape[2]
        return wave

    # Define a function to calculate the Bessel pulse shape
    def _bessel_pulse(self, x, y, t=0):
        first_zero = jn_zeros(self.bessel_order, 1)
        wave = np.zeros([3, x.shape[0], y.shape[1]], order='F')
        shapex = jn(self.bessel_order, (x - (self.offset[1] + self.v[0] * t) * (first_zero[0]) / (self.pulse_width[0] / 2.0)))
        if self.kill_after_first_zero:
            shape_kill = np.abs((x - (self.offset[1] + self.v[0] * t) * (first_zero[0]) / (self.pulse_width[0] / 2.0)) <= (first_zero[0]))
            shapex = shape_kill * shapex
        shapey = self.transversal_function(y)
        shape = shapey * shapex
        wave[0, :, :] = self.amplitude[0] * shape
        wave[1, :, :] = self.amplitude[1] * shape
        wave[2, :, :] = self.amplitude[2] * shape
        return wave

    # Define a function to return a zero wave
    def _off(self, x, y, t=0):
        wave = np.zeros([3, x.shape[0], y.shape[1]], order='F')
        return wave

    # Initialize the Source3D class
    def __init__(self, material, shape='off', **kwargs):
        self._set_f_w(material, kwargs)
        self.options = {}
        self.k = np.asarray([2.0 * np.pi / self.wavelength, 0.0])
        self.v = material.co * np.asarray([1.0 / material.bkg_n[0], 1.0 / material.bkg_n[1])
        self.amplitude = np.asarray([0.0, material.zo, 1.0])
        self.offset = np.zeros([2])
        self.shape = shape
        self.custom = False
        self.function = None
        self.heading = 'x'
        self.averaged = True
        self.cut = False
        self._delta = np.ones([2])
        self._material = material
        self.transversal_shape = 'plane'
        self.transversal_offset = 0.0
        self.transversal_width = 0.0
        self.transversal_function = lambda y: 1.0
        self.transversal_delta = 1.0
        self.num_dim = 2

# Define a subclass 'Source3D' that inherits from 'Sources'
class Source3D(Sources):
    # Define a function to set up source options
    def setup(self, options={}):


   # Transversal Cosine function to calculate wave shape.
    def _transversal_cosine(self, u, v):
        # Extract transversal parameters.
        sv = self.transversal_width[0]
        su = self.transversal_width[1]
        uo = self.transversal_offset[0]
        vo = self.transversal_offset[1]
        
        # Calculate r1 and r2.
        r1 = (u - uo) / sv
        r2 = (v - vo) / su
        
        if self.averaged:
            # Calculate wave shape when averaging is enabled.
            du = self._delta[1]
            dv = self._delta[2]
            ku = np.pi / self.transversal_width[0]
            kv = np.pi / self.transversal_width[1]
            ddu = self._delta[1]
            ddv = self._delta[2]
            shape = (2 * sv * su * np.cos((ku * (x - uo)) / sv) *
                     np.sin((ddu * ku) / sv) * (np.sin((kv * (ddv + y - vo)) / su) +
                                               np.sin((kv * (ddv - y + vo)) / su))) / (du * dv * ku * kv)
        else:
            # Calculate wave shape using a cosine function.
            shape = np.cos(r * np.pi)
        
        # Apply shape constraints.
        shape = shape * (np.abs(r1) <= 0.5) * (np.abs(r2) <= 0.5)

        return shape

    # Transversal Gauss function to calculate wave shape.
    def _transversal_gauss(self, u, p):
        du = self._delta[p]

        # Calculate wave shape using a Gaussian function.
        shape = self.gaussian(u, du, xo=self.transversal_offset[p], s=self.transversal_width[p])

        return shape

    # Transversal Bessel function to calculate wave shape.
    def _transversal_bessel(self, u, p):
        from scipy.special import jn, jn_zeros
        first_zero = jn_zeros(self.transversal_bessel_order, 1)
        r = (u - self.transversal_offset[p]) / (self.transversal_width[p] / 2.0)

        # Calculate wave shape using a Bessel function.
        shape = jn(self.transversal_bessel_order, r * (first_zero[0]))

        if self.transversal_kill_after_first_zero:
            shape_kill = np.abs(r * (first_zero[0])) <= (first_zero[0])
            shape = shape_kill * shape

        return shape

    # Plane wave generation function.
    def _plane(self, x, y, z, t=0):
        # Initialize an array to represent the wave.
        wave = np.zeros([6, x.shape[0], y.shape[1], z.shape[2]], order='F')

        # Calculate harmonic components based on transversal and harmonic functions.
        harmonic = self.transversal_function(y, z) * self.harmonic(x, self.delta[0], xo=self.offset[0], k=self.k[0], omega=self.omega, t=t)

        # Assign amplitudes to the appropriate components.
        wave[1, :, :, :] = self.amplitude[1] * harmonic
        wave[5, :, :, :] = self.amplitude[5] * harmonic

        return wave

    # Pulse wave generation function.
    def _pulse(self, x, y, z, t=0):
        # Initialize an array to represent the wave.
        wave = np.zeros([6, x.shape[0], y.shape[1], z.shape[2]], order='F')

        # Determine the dimensions for heading and polarization.
        dimh = heading[self.heading]
        dimp = polarization[self.heading]

        if t <= self.t_off:
            # Calculate the wave shape within the specified time range.
            grid = [x, y, z]
            shape = 1.0
            for i in range(len(self.heading)):
                h = dimh[i]
                shape = shape * self.gaussian(grid[h], self._delta[h], xo=self.offset[h], s=self.pulse_width[h], v=self.v[h], t=t)
            shape = self.transversal_function(y, z) * shape
        else:
            # Wave is off when t exceeds the specified time.
            shape = 0.0

        # Assign amplitudes to the appropriate components.
        wave[1, :, :, :] = self.amplitude[1] * shape
        wave[5, :, :, :] = self.amplitude[5] * shape

        return wave

    # Harmonic pulse wave generation function.
    def _harmonic_pulse(self, x, y, z, t=0):
        # Initialize an array to represent the wave.
        wave = np.zeros([6, x.shape[0], y.shape[1], z.shape[2]], order='F')

        # Calculate harmonic components for two waveforms.
        harmonic1 = self.harmonic_function(self.k[0] * (x - self.offset[1]) - self.omega * t)
        harmonic2 = self.harmonic_function(self.k[0] * (x - self.offset[5]) - self.omega * t)

        # Calculate wave shape in x direction using a shape function.
        shapex1 = self.transversal_function(y, z) * self.shape_function(-(x - (self.offset[1] + self.v[0] * t))**2 / self.pulse_width**2)
        shapex2 = self.transversal_function(y, z) * self.shape_function(-(x - (self.offset[5] + self.v[0] * t))**2 / self.pulse_width**2)

        # Assign amplitudes to the appropriate components.
        wave[1, :, :, :] = self.amplitude[1] * shapex1 * harmonic1
        wave[5, :, :, :] = self.amplitude[5] * shapex2 * harmonic2

        return wave

    # Bessel pulse wave generation function.
    def _bessel_pulse(self, x, y, z, t=0):
        from scipy.special import jn, jn_zeros
        first_zero = jn_zeros(self.bessel_order, 1)

        # Initialize an array to represent the wave.
        wave = np.zeros([6, x.shape[0], y.shape[1], z.shape[2]], order='F')

        # Calculate wave shape in x direction using Bessel functions.
        shapex1 = self.transversal_function(y, z) * jn(self.bessel_order, (x - (self.offset[1] + self.v[0] * t) * (first_zero[0]) / (self.pulse_width / 2.0)))
        shapex2 = self.transversal_function(y, z) * jn(self.bessel_order, (x - (self.offset[5] + self.v[0] * t) * (first_zero[0]) / (self.pulse_width / 2.0)))

        if self.kill_after_first_zero:
            # Apply a constraint to wave shape based on the first zero of the Bessel function.
            shape_kill1 = np.abs((x - (self.offset[1] + self.v[0] * t) * (first_zero[0]) / (self.pulse_width / 2.0))) <= (first_zero[0])
            shape_kill2 = np.abs((x - (self.offset[5] + self.v[0] * t) * (first_zero[0]) / (self.pulse_width / 2.0))) <= (first_zero[0])

            shapex1 = shape_kill1 * shapex1
            shapex2 = shape_kill2 * shapex2

        # Assign amplitudes to the appropriate components.
        wave[1, :, :, :] = self.amplitude[1] * shapex1
        wave[5, :, :, :] = self.amplitude[5] * shapex2

        return wave

    # Function to generate an "off" wave (no wave).
    def _off(self, x, y, z, t=0):
        # Initialize an array representing an "off" wave.
        wave = np.zeros([6, x.shape[0], y.shape[1], z.shape[2]], order='F')

        return wave
    # Constructor for initializing the WaveGenerator.
    def __init__(self, material, shape='off', **kwargs):
        # Set the material and options for the wave generator.
        self._set_f_w(material, kwargs)
        self.options = {}
        
        # Define wave properties: wavevector, velocity, amplitude, offsets, and more.
        self.k = np.asarray([2.0 * np.pi / self.wavelength, 0.0, 0.0])
        self.v = material.co * np.asarray([1.0 / material.bkg_n[0], 1.0 / material.bkg_n[1], 1.0 / material.bkg_n[2])
        self.amplitude = np.asarray([0.0, material.zo, 0.0, 0.0, 0.0, 1.0])
        self.offset = np.zeros([6])
        
        # Initialize transversal properties.
        self.transversal_shape = 'plane'
        self.transversal_offset = np.zeros([2])
        self.transversal_width = np.zeros([2])
        self.transversal_function = None
        
        # Set the default shape and function for the wave.
        self.shape = shape
        self.custom = False
        self.function = None
        self._material = material
        self._delta = np.zeros([3])
        self.averaged = False
        self.cut = False
  #!/usr/bin/env python
# encoding: utf-8

"""
Wave propagation Riemann solvers implemented in Fortran.
"""

# The above comments provide a brief description of the module.

__pdoc__ = {
    'maxwell_1d': False,
    'maxwell_2d': False,
    'maxwell_3d': False,
    'build': False,
    '__pycache__': False
}

# The __pdoc__ dictionary is used for controlling documentation generation.
# It's set to False for specific functions or modules to exclude them from documentation generation.

# Import statements for Fortran solvers implemented in other modules.
from . import maxwell_1d_nc_rp
from . import maxwell_1d_nc_tfluct
from . import maxwell_1d_rp
from . import maxwell_1d_tfluct

from . import maxwell_2d_nc_rp
from . import maxwell_2d_nc_tfluct
from . import maxwell_2d_rp
from . import maxwell_2d_tfluct

from . import maxwell_3d_nc_rp
from . import maxwell_3d_nc_tfluct

# The code imports various Fortran solvers for 1D, 2D, and 3D problems from other modules.

# It's a good practice to add comments and docstrings to individual functions and classes as needed for clarity.
# Additionally, you can provide more specific comments for each module or function to explain their purpose and usage.

"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""

import numpy as np


# new_utils.py
def scale_data(X, y):
    # Ensure X is floating point and scaled between 0 and 1
    is_scaled = np.all(X >= 0) and np.all(X <= 1)
    is_floating_point = X.dtype in [np.float32, np.float64]
    
    # Ensure y is integer
    is_integer = y.dtype in [np.int32, np.int64]
    
    return is_scaled and is_floating_point and is_integer

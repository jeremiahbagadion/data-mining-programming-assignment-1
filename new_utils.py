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
#def scale_data(X, y):
    # Ensure X is floating point and scaled between 0 and 1
    #is_scaled = np.all(X >= 0) and np.all(X <= 1)
    #is_floating_point = X.dtype in [np.float32, np.float64]
    
    # Ensure y is integer
    #is_integer = y.dtype in [np.int32, np.int64]
    
    #return is_scaled and is_floating_point and is_integer


# Inside new_utils.py
def scale_data(X):
    # Scale X here
    scaled_X = X / X.max()  # Example scaling operation
    return scaled_X, True  # Return scaled data and a boolean indicating success


def print_scores(answer):
    for ntrain, results in answer.items():
        print(f"Training Size: {ntrain}")
        for part, scores in results.items():
            if isinstance(scores, dict):  # For parts C, D, and F
                print(f"\t{part}:")
                for score_type, score_value in scores.items():
                    print(f"\t\t{score_type}: {score_value}")
            else:  # For ntrain, ntest, and class counts
                print(f"\t{part}: {scores}")



def remove_90_percent_nines(X, y):
            mask = y == 9
            remove_nines = np.random.choice(np.where(mask)[0], size=int(np.sum(mask) * 0.9), replace=False)
            keep = np.setdiff1d(np.arange(len(y)), remove_nines)
            return X[keep], y[keep]
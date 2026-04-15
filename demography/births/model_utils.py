""" model_utils.py

Shared helper functions for building design matrices and prediction 
grids across regression scripts (AgeAtKthKid.py, ZeroInflatedNumKids.py).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def build_design_matrix(sf, cat_cols, X_template=None):
    """
    Build a design matrix with categorical dummies, a constant, and 
    a centered year term (read from sf["year_c"]).
    
    If X_template is provided, reindex dummy columns to match it 
    (for prediction grids). Otherwise, build from scratch (for 
    training data).
    """
    X = pd.get_dummies(sf[cat_cols], drop_first=True)
    if X_template is not None:
        X = X.reindex(columns=X_template.columns.drop(["const", "year_c"]), fill_value=0.0)
    X = sm.add_constant(X, has_constant="add")
    X["year_c"] = sf["year_c"].to_numpy()
    return X


def build_prediction_grid(province, sf, year_max, cat_cols, 
                          extra_dims=None, grid_names=None):
    """
    Build a full factorial prediction grid for a province.
    """
    year_min = int(sf["year"].min())
    
    levels = [[province]]
    for c in cat_cols:
        levels.append(sf[c].cat.categories)
    levels.append(np.arange(year_min, year_max + 1))
    
    if extra_dims is not None:
        for col_name, col_values in extra_dims.items():
            levels.append(col_values)
    
    grid = pd.MultiIndex.from_product(
        levels, names=grid_names
    ).to_frame(index=False)
    
    # Center year using training data mean
    grid["year_c"] = grid["year"].astype(float) - sf["year"].astype(float).mean()
    
    return grid
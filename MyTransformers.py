import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing RobustScaler
from sklearn.pipline import FeatureUnion

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_names):
        self._feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Xtemp = X.copy()
        return Xtemp[self._feature_names]
    

    

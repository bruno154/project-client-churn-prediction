import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

class PreProcessingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xtemp = X.copy()
        
        # Ordinal Encoder
        enc = OrdinalEncoder()
        
        # Kmeans Model
        model = KMeans(n_clusters=2, init='k-means++')
        
        # Robust Scaler
        scaler = RobustScaler()
        
        
        
        # Filtrando 'CreditScore' somente o limite inferior
        Xtemp = Xtemp.loc[Xtemp['CreditScore']>400,]
        
        # Filtrando 'Age' somente o limite inferior
        Xtemp = Xtemp.loc[Xtemp['Age']<59, ]

        # Filtrando 'NumOfProducts'
        Xtemp = Xtemp.loc[Xtemp['NumOfProducts']<4, ]
        
        
        
        
        
        #............feature Enginerring.............
        
        
        
        
        
        
        
        # Alterando os tipos Geography e Gender
        varr = ['Geography', 'Gender']
        for var in varr:
            Xtemp[var] = Xtemp[var].astype('category')
        
        # Ordinal Encoder
        variaveis = ['Geography', 
                     'Gender',
                     'kmeans_group',
                     'CreditScore_new', 	
                     'Age_new', 	
                     'Tenure_new', 	
                     'Balance_new', 	
                     'EstimatedSalary_new']
        

        for var in variaveis:
                Xtemp[var] = enc.fit_transform(np.array(Xtemp[var]).reshape(-1,1))

        # RobustScaler
        variaveis = ['CreditScore', 
                     'Age', 
                     'Tenure', 
                     'Balance', 
                     'NumOfProducts', 
                     'EstimatedSalary', 
                     'mean_x', 
                     'mean_y',
                     'LTV']
        
        # Instanciando o Robust Scaler
        scaler = RobustScaler()

        for var in variaveis:
            Xtemp[var] = scaler.fit_transform(np.array(Xtemp[var]).reshape(-1,1))

        # Instanciando o blanciador
#         smt = SMOTETomek(sampling_strategy='minority' ,random_state=42)
#         Xtemp_smt, _ = smt.fit_resample(Xtemp, y)

        return Xtemp

    

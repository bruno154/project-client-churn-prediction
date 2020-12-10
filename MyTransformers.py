import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.cluster import KMeans
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
        
        Xtemp.drop(['RowNumber','CustomerId','Surname'], inplace=True, axis=1)
        
        Xtemp_temp = Xtemp.copy()
        
        # Encoder
        Xtemp_temp['Gender'] = Xtemp_temp['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        Xtemp_temp['Geography'] = enc.fit_transform(np.array(Xtemp_temp['Geography']).reshape(-1,1))
        
        # kmeans model
        model = KMeans(n_clusters=2,init='k-means++')
        model.fit(Xtemp_temp)
        Xtemp['kmeans_group'] = model.labels_
        Xtemp['kmeans_group'] = Xtemp['kmeans_group'].astype('category')
        Xtemp['kmeans_group'] = Xtemp['kmeans_group'].apply(lambda x: 'G1' if x==1 else 'G2')
        
        # Criar variável Balance por location
        group_balance = Xtemp.groupby('Geography').agg({'Balance': ['mean']}).reset_index()
        group = pd.concat([group_balance['Geography'],group_balance['Balance']['mean']], axis=1)
        Xtemp = Xtemp.merge(group, left_on='Geography', right_on='Geography', how='inner')
        
        # Criar variável EstimatedSalary por location
        group_balance = Xtemp.groupby('Geography').agg({'EstimatedSalary': ['mean']}).reset_index()
        group = pd.concat([group_balance['Geography'],  group_balance['EstimatedSalary']['mean']], axis=1)
        Xtemp = Xtemp.merge(group, left_on='Geography', right_on='Geography', how='inner')
        
        # Binning
        # CreditScore
        Xtemp['CreditScore_new'] = pd.qcut(Xtemp['CreditScore'],q=10, duplicates='drop')

        # Age
        Xtemp['Age_new'] = pd.qcut(Xtemp['Age'],q=10, duplicates='drop')

        # Tenure
        Xtemp['Tenure_new'] = pd.qcut(Xtemp['Tenure'],q=10, duplicates='drop')

        # Balance
        Xtemp['Balance_new'] = pd.qcut(Xtemp['Balance'],q=[.35, .70, 1], duplicates='drop')

        # EstimatedSalary
        Xtemp['EstimatedSalary_new'] = pd.qcut(Xtemp['EstimatedSalary'],q=10, duplicates='drop')
        
        # LTV
        balance = Xtemp['Balance'].astype('int64')
        Xtemp['LTV_bruno'] = balance / (Xtemp['Tenure'] + 0.1)

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
                     'LTV_bruno']
        
        # Instanciando o Robust Scaler
        scaler = RobustScaler()

        for var in variaveis:
            Xtemp[var] = scaler.fit_transform(np.array(Xtemp[var]).reshape(-1,1))
            
        Xtemp = Xtemp[[  
                         'Balance_new',                  
                         'Gender',                       
                         'IsActiveMember',               
                         'Geography',                    
                         'EstimatedSalary',              
                         'Age_new',                      
                         'EstimatedSalary_new',          
                         'mean_x',                       
                         'Age',                          
                         'NumOfProducts',                
                         'kmeans_group',                 
                         'Balance',                      
                         'Tenure_new',                   
                         'Tenure',                       
                         'CreditScore_new',
                         'CreditScore',
                         'mean_y',                       
                         'LTV_bruno'
                                     ]]

        # Instanciando o blanciador
#         smt = SMOTETomek(sampling_strategy='minority' ,random_state=42)
#         Xtemp_smt, _ = smt.fit_resample(Xtemp, y)

        return Xtemp

    

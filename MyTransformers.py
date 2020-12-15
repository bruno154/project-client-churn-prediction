import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
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
        
        # Instanciando o PowerTransformer
        scaler = PowerTransformer()
        
        # Filtrando 'CreditScore' somente o limite inferior
        Xtemp = Xtemp.loc[Xtemp['CreditScore']>400,]
        
        # Filtrando 'Age' somente o limite inferior
        Xtemp = Xtemp.loc[Xtemp['Age']<59, ]

        # Filtrando 'NumOfProducts'
        Xtemp = Xtemp.loc[Xtemp['NumOfProducts']<4, ]
        
        # Eliminando os Indentificadores unicos.
        Xtemp.drop(['RowNumber','CustomerId','Surname'], inplace=True, axis=1)
        
        #Xtemp_temp = Xtemp.copy()
        
        # Encoder
        Xtemp['Gender'] = Xtemp['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        Xtemp['Geography'] = enc.fit_transform(np.array(Xtemp['Geography']).reshape(-1,1))
        
        # kmeans model
        model = KMeans(n_clusters=2,init='k-means++')
        model.fit(Xtemp)
        Xtemp['kmeans_group'] = model.labels_
        Xtemp['kmeans_group'] = Xtemp['kmeans_group'].astype('category')
        Xtemp['kmeans_group'] = Xtemp['kmeans_group'].apply(lambda x: 'G1' if x==1 else 'G2')
        
        # Criar variável Balance por location
        group_balance = Xtemp.groupby('Geography').agg({'Balance': ['mean']}).reset_index()
        group = pd.concat([group_balance['Geography'],group_balance['Balance']['mean']], axis=1)
        Xtemp = Xtemp.merge(group, left_on='Geography', right_on='Geography', how='inner')
        
        # Criar variável EstimatedSalary por location
        group_EstimatedSalary = Xtemp.groupby('Geography').agg({'EstimatedSalary': ['mean']}).reset_index()
        group = pd.concat([group_EstimatedSalary['Geography'],  group_EstimatedSalary['EstimatedSalary']['mean']], axis=1)
        Xtemp = Xtemp.merge(group, left_on='Geography', right_on='Geography', how='inner')
        
        # Criar variável EstimatedSalary por gender
        group_Gender = Xtemp.groupby('Gender').agg({'EstimatedSalary': ['mean']}).reset_index()
        group = pd.concat([group_Gender['Gender'],  group_Gender['EstimatedSalary']['mean']], axis=1)
        group = group.rename(columns={"mean":"EstimatedSalary_mean_gender"})
        Xtemp = Xtemp.merge(group, left_on='Gender', right_on='Gender', how='inner')
        
        # Criar variável EstimatedSalary por hascrcard
        group_HasCrCard = Xtemp.groupby('HasCrCard').agg({'EstimatedSalary': ['mean']}).reset_index()
        group = pd.concat([group_HasCrCard['HasCrCard'],  group_HasCrCard['EstimatedSalary']['mean']], axis=1)
        group = group.rename(columns={"mean":"group_HasCrCard_mean"})
        Xtemp = Xtemp.merge(group, left_on='HasCrCard', right_on='HasCrCard', how='inner')
        
        # Criar variável CreditScore score por Hascrcard
        group_hascrcard = Xtemp.groupby('HasCrCard').agg({'CreditScore': ['mean']}).reset_index()
        group = pd.concat([group_hascrcard['HasCrCard'],  group_hascrcard['CreditScore']['mean']], axis=1)
        group = group.rename(columns={"mean":"hascrcard_mean_credit"})
        Xtemp = Xtemp.merge(group, left_on='HasCrCard', right_on='HasCrCard', how='inner')
        
        # Criar variável CreditScore score por Gender
        group_gender = Xtemp.groupby('Gender').agg({'CreditScore': ['mean']}).reset_index()
        group = pd.concat([group_gender['Gender'],  group_gender['CreditScore']['mean']], axis=1)
        group = group.rename(columns={"mean":"gender_mean_credit"})
        Xtemp = Xtemp.merge(group, left_on='Gender', right_on='Gender', how='inner')

        # LTV
        balance = Xtemp['Balance'].astype('int64')
        Xtemp['LTV'] = balance / (Xtemp['Tenure'] + 0.1)

        # Alterando os tipos Geography e Gender
        varr = ['Geography', 'Gender', 'IsActiveMember']
        for var in varr:
            Xtemp[var] = Xtemp[var].astype('category')
        
        # Ordinal Encoder
        variaveis_category = Xtemp.select_dtypes('category')

        for var in variaveis_category:
                Xtemp[var] = enc.fit_transform(np.array(Xtemp[var]).reshape(-1,1))

        # Numerical var
        variaveis_numerical = Xtemp.select_dtypes(['int64','float64'])

        for var in variaveis_numerical:
            Xtemp[var] = scaler.fit_transform(np.array(Xtemp[var]).reshape(-1,1))
            
        Xtemp = Xtemp[ ['EstimatedSalary', 
                        'Geography', 
                        'Age', 
                        'NumOfProducts', 
                        'Tenure', 
                        'Gender', 
                        'IsActiveMember', 
                        'Balance']]


        return Xtemp

    

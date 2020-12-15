# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency
from scikitplot.metrics import plot_confusion_matrix, plot_roc

def multi_boxplots(df, variables: list) -> None:

    """
    Function to check for outliers visually through a boxplot

    data: DataFrame

    variable: list of numerical variables
    """

    # set of initial plot posistion
    n = 1

    plt.figure(figsize=(18, 10))
    for column in df[variables].columns:
        plt.subplot(3, 3, n)
        _ = sns.boxplot(x=column, data=df)
        n += 1

    plt.subplots_adjust(hspace=0.3)

    plt.show()
    
def Myheat_map(dataset, variaveis):
    
    """
    
    
    """

    df_corr = dataset[variaveis].corr()

    fig, ax = plt.subplots(figsize=(16, 10))
    # mask
    mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
    # adjust mask and df
    mask = mask[1:, :-1]
    corr = df_corr.iloc[1:,:-1].copy()
    # color map
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

    # plot heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                   linewidths=5, cmap=cmap, vmin=-1, vmax=1, 
                   cbar_kws={"shrink": .8}, square=True)
    yticks = [i.upper() for i in corr.index]
    xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
    plt.xticks(plt.xticks()[0], labels=xticks, rotation=20)

    # title
    title = 'CORRELATION MATRIX\n'
    plt.title(title, loc='left', fontsize=18)
    plt.show()

def cramer_v(var_x, var_y):
    """
    Function to calculate the Cramers v correlation.

    """
    # builds contigency matrix (or confusion matrix)
    confusion_matrix_v = pd.crosstab(var_x, var_y).values

    # gets the sum of all values in the matrix
    n = confusion_matrix_v.sum()

    # gets the rows, cols
    r, k = confusion_matrix_v.shape

    # gets the chi-squared
    chi2 = chi2_contingency(confusion_matrix_v)[0]

    # makes the bias correction
    chi2corr = max(0, chi2 - (k-1) * (r-1) / (n-1))
    kcorr = k - (k-1) ** 2 / (n-1)
    rcorr = r - (r-1) ** 2 / (n-1)

    # returns cramér V
    return np.sqrt((chi2corr/n) / min(kcorr-1, rcorr-1))

def hipo_test(*samples):

    samples = samples

    try:
        if len(samples) == 2:
            stat, p = ttest_ind(*samples)
        elif len(samples) > 2:
            stat, p = f_oneway(*samples)
    except:
        raise Exception("Deve ser fornecido pelo menos duas samples!!!")

    if p < 0.05:
        print(f'O valor de p é: {p}')
        print('Provável haver diferença')
    else:
        print(f'O valor de p é: {p}')
        print('Provável que não haja diferença')

    return stat, p

def learning_curves(estimator, features, target, train_sizes, cv):
    
    """
    
    """

    train_sizes, train_scores, validation_scores = learning_curve(
    estimator, features, target, train_sizes =
    train_sizes,
    cv = cv, scoring = 'recall')
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

    plt.ylabel('Recall', fontsize = 12)
    plt.xlabel('Training set size', fontsize = 12)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 14, y = 1.03)
    plt.legend()

def model_selection(Xtrain, ytrain, Xtest, ytest):
    """
    
    """
    
    # Modelos
    models = [('lr',LogisticRegression(),1),
              ('svm',SVC(),2),
              ('lda',LinearDiscriminantAnalysis(),3),
              ('qda',QuadraticDiscriminantAnalysis(),4),
              ('dt',DecisionTreeClassifier(),5),
              ('rf', RandomForestClassifier(class_weight="balanced"),6),
              ('lgb',lgb.LGBMClassifier(),7),
              ('xgboost', XGBClassifier(),8)]

    # Resultados
    resultados = {'LR': [],
                  'SVM': [],
                  'LDA': [],
                  'QDA': [],
                  'DecisionTree': [],
                  'RandomForestClassifier': [],
                  'LGBM': [],
                  'XGBOOST': []}

    # Testando algoritmos
    for name, model,_ in models:
        
        counter = 0
        resultado = []
        while counter <= 10:
        
            # resultado
            model.fit(Xtrain, ytrain)
            pred = model.predict(Xtest)
            score = recall_score(ytest, pred)
            resultado.append(score)
            counter += 1


        if name == 'lr':
            resultados['LR'].append(np.mean(resultado))
        elif name == 'knn':
            resultados['KNN'].append(np.mean(resultado)) 
        elif name == 'svm':
            resultados['SVM'].append(np.mean(resultado))
        elif name == 'dt':
            resultados['DecisionTree'].append(np.mean(resultado))
        elif name == 'rf':
            resultados['RandomForestClassifier'].append(np.mean(resultado))
        elif name == 'lgb':
            resultados['LGBM'].append(np.mean(resultado))
        elif name == 'xgboost':
            resultados['XGBOOST'].append(np.mean(resultado))
        elif name =='lda':
            resultados['LDA'].append(np.mean(resultado))
        elif name == 'qda':
            resultados['QDA'].append(np.mean(resultado))
            

    # Painel
    resultados_df = pd.DataFrame(resultados)
    return resultados_df

def train_SVM(Xtrain, ytrain, Xtest, ytest, C, kernel, degree, class_weight="balanced", probability=True):

    model = SVC(C=C,
                kernel=kernel,
                degree=degree,
                class_weight=class_weight,
                probability=probability)

    # Dicionário de metricas
    resultados = {'ACC': [],
                  'KAPPA': [],
                  'RECALL': [],
                  'F1': [],
                  'PRECISION': []}

    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtest)
    proba = model.predict_proba(Xtest)
    Acc = accuracy_score(ytest, pred)
    Kappa =  cohen_kappa_score(ytest, pred)
    Recall = recall_score(ytest, pred)
    F1 = f1_score(ytest,pred)
    Precision = precision_score(ytest, pred)

    resultados['ACC'].append(Acc)
    resultados['KAPPA'].append(Kappa)
    resultados['RECALL'].append(Recall)
    resultados['F1'].append(F1)
    resultados['PRECISION'].append(Precision)
    
    # Salvando o modelo em pickle
    with open('models/modelo_SVM.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Painel
    painel_df = pd.DataFrame(resultados).T
    painel_df.rename(columns={0:"SVM Metrics"}, inplace=True)
    
    # Confusion Matrix e ROC Curve
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
    plot_confusion_matrix(ytest, pred, normalize=True, ax=ax1)
    plot_roc(ytest, proba, ax=ax2)
    
    return painel_df, model

def train_SVM(Xtrain, ytrain, Xtest, ytest, C, kernel, degree, class_weight="balanced", probability=True):

    model = SVC(C=C,
                kernel=kernel,
                degree=degree,
                class_weight=class_weight,
                probability=probability)

    # Dicionário de metricas
    resultados = {'ACC': [],
                  'KAPPA': [],
                  'RECALL': [],
                  'F1': [],
                  'PRECISION': []}

    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtest)
    proba = model.predict_proba(Xtest)
    Acc = accuracy_score(ytest, pred)
    Kappa =  cohen_kappa_score(ytest, pred)
    Recall = recall_score(ytest, pred)
    F1 = f1_score(ytest,pred)
    Precision = precision_score(ytest, pred)

    resultados['ACC'].append(Acc)
    resultados['KAPPA'].append(Kappa)
    resultados['RECALL'].append(Recall)
    resultados['F1'].append(F1)
    resultados['PRECISION'].append(Precision)
    
    # Salvando o modelo em pickle
    with open('models/modelo_SVM.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Painel
    painel_df = pd.DataFrame(resultados).T
    painel_df.rename(columns={0:"SVM Metrics"}, inplace=True)
    
    # Confusion Matrix e ROC Curve
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
    plot_confusion_matrix(ytest, pred, normalize=True, ax=ax1)
    plot_roc(ytest, proba, ax=ax2)
    
    return painel_df, model

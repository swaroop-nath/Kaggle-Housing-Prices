import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import r2_score, mean_squared_error as mse

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer

from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

import csv

# Loading the dataset
data = pd.read_csv('train.csv')

def dampenSkew(df, num_features):
    skew_matrix = df[num_features].apply(lambda column: skew(column)).sort_values(ascending = False)
    skewed_features = skew_matrix[(abs(skew_matrix) > 1.0)].index
    for feature in skewed_features:
        df[feature] = boxcox1p(df[feature], boxcox_normmax(df[feature] + 1))
    return df

def preProcessData(dataset):
    dataset['Alley'].replace(np.nan, 'NA', inplace = True)
    dataset['MasVnrType'].replace(np.nan, dataset['MasVnrType'].unique()[0], inplace = True)
    dataset['BsmtQual'].replace(np.nan, 'NA', inplace = True)
    dataset['BsmtCond'].replace(np.nan, 'NA', inplace = True)
    dataset['BsmtExposure'].replace(np.nan, 'NA', inplace = True)
    dataset['BsmtFinType1'].replace(np.nan, 'NA', inplace = True)
    dataset['BsmtFinType2'].replace(np.nan, 'NA', inplace = True)
    dataset['Electrical'].replace(np.nan, dataset['Electrical'].unique()[0], inplace = True)
    dataset['FireplaceQu'].replace(np.nan, 'NA', inplace = True)
    dataset['GarageType'].replace(np.nan, 'NA', inplace = True)
    dataset['GarageFinish'].replace(np.nan, 'NA', inplace = True)
    dataset['GarageQual'].replace(np.nan, 'NA', inplace = True)
    dataset['GarageCond'].replace(np.nan, 'NA', inplace = True)
    dataset['PoolQC'].replace(np.nan, 'NA', inplace = True)
    dataset['Fence'].replace(np.nan, 'NA', inplace = True)
    dataset['MiscFeature'].replace(np.nan, 'NA', inplace = True)
    
    dataset = dataset[(dataset.TotalBsmtSF < 5000)]
    
    # Dealing with ordinal categorical variables
    ordinal_cols = {'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                    'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                    'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                    'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
                    'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                    'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
                    }
    
    dataset.replace(ordinal_cols, inplace= True)
    
    # Getting the indices of numeric and categorical features.
    num_feats = list(dataset.dtypes[dataset.dtypes != object].index)
    cat_feats = list(dataset.dtypes[dataset.dtypes == object].index)
    
    num_feats.pop(0) # Getting rid of the ID field.
    num_feats.pop(-1) # Removing SalePrice from the feature matrix
    
    X_num = dataset.loc[:, num_feats]
    X_cat = dataset.loc[:, cat_feats]
    y = dataset.iloc[:, -1]
    y = np.log(y)
    
    X_num = dampenSkew(X_num.copy(), num_feats)
    
    # Cleaning the dataset
    a = pd.isnull(X_num).sum() > 0
    
    # The numeric columns which have NAN values.
    nanColumns = [i for i in a.index.tolist() if a[i] == True]
    
    # Imputing the unknown values
    imputer = Imputer()
    for i in range(len(nanColumns)):
        X_num[nanColumns[i]] = imputer.fit_transform(X_num[nanColumns[i]].values.reshape(-1,1))
    
    # Handling categorical variables
    labelEncoders = []
        
    for x in range(len(cat_feats)):
        labelEncoders.append(LabelEncoder())
        
    oneHotEncoding = OneHotEncoder(categories='auto', drop = 'first')
    
    for i in range(len(cat_feats)):
        print(cat_feats[i])
        X_cat.loc[:, cat_feats[i]] = labelEncoders[i].fit_transform(X_cat.loc[:, cat_feats[i]])
    
    oneHotEncoding = oneHotEncoding.fit(X_cat)
    X_cat = oneHotEncoding.transform(X_cat).toarray()
    
    # Feature Engineering
    X_num['FractionBsmtFinSF1'] = X_num['BsmtFinSF1']/X_num['TotalBsmtSF']
    X_num['FractionBsmtFinSF1'].fillna(0, inplace = True)
    X_num['FractionBsmtFinSF2'] = X_num['BsmtFinSF2']/X_num['TotalBsmtSF']
    X_num['FractionBsmtFinSF2'].fillna(0, inplace = True)
    X_num['FractionBsmtUnfSF'] = X_num['BsmtUnfSF']/X_num['TotalBsmtSF']
    X_num['FractionBsmtUnfSF'].fillna(0, inplace = True)
    
    X_num['FlrSF'] = np.log(X_num['1stFlrSF'] + X_num['2ndFlrSF'])
    
    X_num['IsGoodFinTot'] = X_num['LowQualFinSF'].apply(lambda lowQual: 1 if lowQual == 0 else 0)
    
    X_num['convKitchenInfo'] = X_num.KitchenQual * X_num.KitchenAbvGr
    
    drop_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'KitchenAbvGr', 'KitchenQual']
    X_num = X_num.drop(drop_cols, axis = 1)
    
    X = np.concatenate((X_num.values, X_cat), axis = 1)
    
    return X, y, X_num 

# Preprocessing Data
X, y, X_num = preProcessData(data.copy());

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = False, stratify = None)

# Making the model
params = {
        'n_estimators' : 250, # Boosting parameter
        'learning_rate' : 0.05, # Boosting parameter
        'min_samples_split': 0.05, # Tree parameter
        'min_weight_fraction_leaf': 0.025, # Tree parameter
        'max_depth': 5, # Tree parameter
        'max_features': 'sqrt', # Tree parameter
        'loss' : 'ls'# Misc parameter
        }

regressor = GBR(**params) 
regressor.fit(X_train, y_train)

r2_train = r2_score(y_train, regressor.predict(X_train))
r2_test = r2_score(y_test, regressor.predict(X_test))
mse_train = mse(y_train, regressor.predict(X_train))
mse_test = mse(y_test, regressor.predict(X_test))

#y_pred = regressor.predict(X_test)
cv_score = cross_val_score(regressor, X_test, y_test, cv = 10, scoring = 'neg_mean_squared_error')
print(np.mean(cv_score))

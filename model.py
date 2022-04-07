import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.stats as stats
import wrangle_zillow

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE

import warnings
warnings.filterwarnings("ignore")

def add_features(train, validate, test):
    
    datasets = [train, validate, test]
    
    for dataset in datasets:
        dataset['age'] = 2017 - dataset["yearbuilt"]
        dataset['bathroom_bin'] = pd.qcut(dataset["bathroom"], 2, labels=["fewer_baths","more_baths"])
        dataset['bedroom_bin'] = pd.qcut(dataset["bedroom"], 3, labels = ["low_bedrooms","medium_bedrooms","high_bedrooms"])
        dataset['fullbathcnt_bin'] = pd.cut(dataset.fullbathcnt, 3, labels = ["low","medium","high"])
        dataset['age_bin']=pd.qcut(dataset["age"], 2, labels=['young','old'])
        dataset['has_garage'] = dataset["garage"]>0
        dataset['home_size'] = pd.qcut(dataset["square_feet"],3, labels=['small','medium','large'])
        dataset['living_space'] = dataset["square_feet"] - (40*dataset["bathroom"]+200*dataset["bedroom"])
        dataset['tax_rate'] = dataset["taxamount"]/dataset["tax_value"]
        dataset['acres'] = dataset["lot_size"]/43560
        dataset['structure_dollar_per_sqft'] = dataset["tax_value"]/dataset["square_feet"]
        dataset['land_dollar_per_sqft'] = dataset["landtaxvaluedollarcnt"] / dataset["square_feet"]
        # Sometimes num bathrooms is 0 -> set value to avoid inf/NaN creation
        dataset['bed_bath_ratio'] = np.select([dataset["bathroom"]>0,dataset["bathroom"]==0],[dataset["bedroom"]/dataset["bathroom"],0])
        dataset['is_la'] = dataset["county"] == 'Los Angeles County'
        dataset['is_orange'] = dataset["county"] == 'Orange County'
        dataset['abs_logerror'] = abs(dataset["logerror"])
        dataset['tax_value_bin'] = pd.qcut(dataset["tax_value"], 4, labels=["cheap","medium","high","ultimate_luxury"])
        dataset['lot_size_bin'] = pd.qcut(dataset["lot_size"], 4, labels=["tiny","small","medium","big"])
        # dataset['condition_bin'] = pd.qcut(dataset["condition"], 4, labels=["poor","good","great","best"])
        dataset['bed_to_bath_bin'] = pd.qcut(dataset["bed_bath_ratio"], 3, labels=["low","medium","high"])
        dataset['structure_value_bin'] = pd.qcut(dataset["structure_dollar_per_sqft"], 4, labels=["low","medium","high","very_high"])
        dataset['land_value_bin'] = pd.qcut(dataset["land_dollar_per_sqft"], 4, labels=["low","medium","high","very_high"])
        dataset['has_old_heat'] = dataset["heatingorsystemdesc"] == 'Floor/Wall'
        dataset['taxdelinquencyflag'] = np.where(dataset.taxdelinquencyflag=='Y',1,0)
        dataset["delinquent_years"] = (17-dataset.taxdelinquencyyear)
        # dataset = dataset.drop(columns = ['taxdelinquencyflag'])

    return train, validate, test

def scale_data(train, validate, test, features_to_scale):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Fit the scaler to train data only
    scaler = MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1)

    return train_scaled, validate_scaled, test_scaled

def perform_cluster_analysis(train_scaled, validate_scaled, test_scaled, features_to_cluster, cluster_qty = 3, alpha = 0.05, plots=True):
    
    X = train_scaled[features_to_cluster]
    X_validate = validate_scaled[features_to_cluster]
    X_test = test_scaled[features_to_cluster]
    
    # Name the cluster so different clusters can be compared in same dataframe
    cluster_name = "_".join([feat[:3] for feat in features_to_cluster])
    cluster_name = cluster_name+"_cluster"
    
    if plots:
        # plot elbow chart
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(9, 6))
            pd.Series({k: KMeans(k, random_state=123).fit(X).inertia_ for k in range(2, 18)}).plot(marker='x')
            plt.xticks(range(2, 12))
            plt.xlabel('k')
            plt.ylabel('inertia')
            plt.title(f'Change in inertia as k increases for cluster: {cluster_name}')
            plt.show()
        
    kmeans = KMeans(n_clusters=cluster_qty, random_state=123)
    kmeans.fit(X)

    train_scaled[cluster_name] = kmeans.predict(X)
    train_scaled[cluster_name] = train_scaled[cluster_name].astype('category')
    
    validate_scaled[cluster_name] = kmeans.predict(X_validate)
    validate_scaled[cluster_name] = validate_scaled[cluster_name].astype('category')

    test_scaled[cluster_name] = kmeans.predict(X_test)
    test_scaled[cluster_name] = test_scaled[cluster_name].astype('category')


    if plots:
        sns.barplot(data = train_scaled, x = cluster_name, y='abs_logerror')
        plt.title(f"Mean Absolute Log Error by Cluster for cluster: {cluster_name}")
        plt.ylabel("Mean Absolute Log Error", fontsize=16)
        plt.xlabel("Cluster", fontsize=16)
    
    # Run significance test (t-test)
    overall_mean = train_scaled.abs_logerror.mean()
    # print(f"Overall mean logerror: {overall_mean}")
    for col in set(train_scaled[cluster_name]):
        sample = train_scaled[train_scaled[cluster_name]==col]
        t, p = stats.ttest_1samp(sample.abs_logerror, overall_mean)
        if plots:
            print(col, "Significant? ", p<alpha, "t value: ", t)
        
    return train_scaled, validate_scaled, test_scaled

def plot_bar_comparison(train_scaled, feature_to_cluster, cluster_title, cluster_names):
    
    sns.barplot(data = train_scaled, x = cluster_name, y='abs_logerror')
    plt.title(f"Mean Absolute Log Error by Cluster for {cluster_title}")
    plt.ylabel("Mean Absolute Log Error", fontsize=16)
    plt.xlabel("Cluster", fontsize=16)
    
    

def data_scaling(train, validate, test, to_dummy, features_to_scale, columns_to_use):
    """ Performs scaling and creates dummy variables. Performs operations on all three inputed data sets. Requires lists of features to encode (dummy), features to scale, and columns (features) to input to the feature elimination. """
    
    # Select only the columns will be using in the model going forward
    X_train = train[columns_to_use]
    X_validate = validate[columns_to_use]
    X_test = test[columns_to_use]
    
    # Get the dummy variables for desired features
    X_train_dummies = pd.get_dummies(train[to_dummy], drop_first=True)
    X_validate_dummies = pd.get_dummies(validate[to_dummy], drop_first=True)
    X_test_dummies = pd.get_dummies(test[to_dummy], drop_first=True)
    
    # Scale train, validate, and test sets using the scale_data function from wrangle.pu
    X_train_scaled, X_validate_scaled, X_test_scaled = scale_data(X_train, X_validate, X_test,features_to_scale)
    
    # Concatenate dummy variable df onto scaled df
    X_train_scaled = pd.concat([X_train_scaled, X_train_dummies], axis=1)
    X_validate_scaled = pd.concat([X_validate_scaled, X_validate_dummies], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, X_test_dummies], axis=1)
    
    # Drop original non scaled columns
    X_train_scaled = X_train_scaled.drop(columns = features_to_scale)
    X_validate_scaled = X_validate_scaled.drop(columns = features_to_scale)
    X_test_scaled = X_test_scaled.drop(columns = features_to_scale)


    # Set up the dependent variable in a datafrane
    y_train = train[['logerror']]
    y_validate = validate[['logerror']]
    y_test = test[['logerror']]
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test
    
def model_feature_selection(X_train_scaled, y_train, top_n=3):
    """ Performs scaling and feature selection using recursive feature elimination. Performs operations on all three inputed data sets. Requires lists of features to encode (dummy), features to scale, and columns (features) to input to the feature elimination. """
    
    # Perform Feature Selection using Recursive Feature Elimination
    # Initialize ML algorithm
    lm = LinearRegression()
    # create RFE object - selects top 3 features only
    rfe = RFE(lm, n_features_to_select=top_n)
    # fit the data using RFE
    rfe.fit(X_train_scaled, y_train)
    # get mask of columns selected
    feature_mask = rfe.support_
    # get list of column names
    rfe_features = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    # view list of columns and their ranking

    # get the ranks
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X_train_scaled.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    # sort the df by rank
    rfe_ranks_df.sort_values('Rank')

    return rfe_features
    
def model(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, rfe_features, show_test = False, print_results = True):
    """ Fits data to different regression algorithms and evaluates on validate (and test if desired for final product). Outputs metrics for each algorithm (r2, rmse) as a Pandas DataFrame. """
    
    y_train['log_error'] = y_train['logerror']
    y_validate['log_error'] = y_validate['logerror']
    y_test['log_error'] = y_test['logerror']

    # Just using rfe features
    X_train_scaled = X_train_scaled[rfe_features]
    X_validate_scaled = X_validate_scaled[rfe_features]
    X_test_scaled = X_test_scaled[rfe_features]
    
    print(f"Using {X_train_scaled.columns}")
    ### BASELINE
    
    # 1. Predict log_error_pred_mean
    log_error_pred_mean = y_train['log_error'].mean()
    y_train['log_error_pred_mean'] = log_error_pred_mean
    y_validate['log_error_pred_mean'] = log_error_pred_mean

    # 2. compute log_error_pred_median
    log_error_pred_median = y_train['log_error'].median()
    y_train['log_error_pred_median'] = log_error_pred_median
    y_validate['log_error_pred_median'] = log_error_pred_median

    # 3. RMSE of log_error_pred_mean
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_mean)**(1/2)
    if print_results:
        print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of log_error_pred_median
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_median)**(1/2)

    if print_results:

        print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    ### OLS Linear Regression
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_scaled, y_train.log_error)

    # predict train
    y_train['log_error_pred_lm'] = lm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_lm)**(1/2)

    # predict validate
    y_validate['log_error_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_lm)**(1/2)

    if print_results:
        print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    # predict test
    if show_test:
        
        y_test['log_error_pred_lm'] = lm.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.log_error, y_test.log_error_pred_lm)**(1/2)

    # Lasso-Lars
    
    # create the model object
    lars = LassoLars(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train_scaled, y_train.log_error)

    # predict train
    y_train['log_error_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_lars)**(1/2)

    # predict validate
    y_validate['log_error_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_lars)**(1/2)

    if print_results:

        print("RMSE for OLS using LarsLasso\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)

    # predict test
    if show_test:
        
        y_test['log_error_pred_lars'] = lars.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.log_error, y_test.log_error_pred_lars)**(1/2)

    # Tweedie
    
    # create the model object
    glm = TweedieRegressor(power=0, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_scaled, y_train.log_error)

    # predict train
    y_train['log_error_pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_glm)**(1/2)

    # predict validate
    y_validate['log_error_pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_glm)**(1/2)
    
    if print_results:
        print("RMSE for GLM using Tweedie, power=0 & alpha=0\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
        
    # predict test
    if show_test:
        
        y_test['log_error_pred_glm'] = glm.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.log_error, y_test.log_error_pred_glm)**(1/2)

    # Polynomial features
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2,interaction_only=False)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.log_error)

    # predict train
    y_train['log_error_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_error_pred_lm2)**(1/2)

    # predict validate
    y_validate['log_error_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_error_pred_lm2)**(1/2)
    
    if print_results:
        print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
        
    # predict test
    if show_test:
        
        y_test['log_error_pred_lm2'] = lm2.predict(X_test_degree2)
        rmse_test = mean_squared_error(y_test.log_error, y_test.log_error_pred_lm2)**(1/2)

    results = pd.concat([
        y_train.apply(lambda col: r2_score(y_train.log_error, col)).rename('r2_train'),
        y_train.apply(lambda col: mean_squared_error(y_train.log_error, col)).rename('mse_train'),
        y_validate.apply(lambda col: r2_score(y_validate.log_error, col)).rename('r2_validate'),
        y_validate.apply(lambda col: mean_squared_error(y_validate.log_error, col)).rename('mse_validate')
        ], axis=1).assign(
            rmse_validate=lambda df: df.mse_validate.apply(lambda x: x**0.5)
        )
        
    results = results.assign(rmse_train= lambda results: results.mse_train.apply(lambda x: x**0.5))

    if show_test:
        results = pd.concat([
        y_train.apply(lambda col: r2_score(y_train.log_error, col)).rename('r2_train'),
        y_train.apply(lambda col: mean_squared_error(y_train.log_error, col)).rename('mse_train'),
        y_validate.apply(lambda col: r2_score(y_validate.log_error, col)).rename('r2_validate'),
        y_validate.apply(lambda col: mean_squared_error(y_validate.log_error, col)).rename('mse_validate'),
        y_test.apply(lambda col: r2_score(y_test.log_error, col)).rename('r2_test'),
        y_test.apply(lambda col: mean_squared_error(y_test.log_error, col)).rename('mse_test'),
        ], axis=1).assign(
            rmse_validate=lambda df: df.mse_validate.apply(lambda x: x**0.5)
        )
        
        results = results.assign(rmse_train= lambda results: results.mse_train.apply(lambda x: x**0.5))
        results = results.assign(rmse_test= lambda results: results.mse_test.apply(lambda x: x**0.5))
    
    return results


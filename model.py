import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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
        dataset['abs_logerror'] = abs(dataset["logerror"])
        dataset['tax_value_bin'] = pd.qcut(dataset["tax_value"], 4, labels=["cheap","medium","high","ultimate_luxury"])
        dataset['lot_size_bin'] = pd.qcut(dataset["lot_size"], 4, labels=["tiny","small","medium","big"])
        # dataset['condition_bin'] = pd.qcut(dataset["condition"], 4, labels=["poor","good","great","best"])
        dataset['bed_to_bath_bin'] = pd.qcut(dataset["bed_bath_ratio"], 3, labels=["low","medium","high"])
        dataset['structure_value_bin'] = pd.qcut(dataset["structure_dollar_per_sqft"], 4, labels=["low","medium","high","very_high"])
        dataset['land_value_bin'] = pd.qcut(dataset["land_dollar_per_sqft"], 4, labels=["low","medium","high","very_high"])
        dataset['has_old_heat'] = dataset["heatingorsystemdesc"] == 'Floor/Wall'

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

def perform_cluster_analysis(train_scaled, features_to_cluster, cluster_qty = 3, alpha = 0.05):
    
    X = train_scaled[features_to_cluster]
    
    # Name the cluster so different clusters can be compared in same dataframe
    cluster_name = "_".join([feat[:3] for feat in features_to_cluster])
    cluster_name = cluster_name+"_cluster"
    
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
    
    sns.barplot(data = train_scaled, x = cluster_name, y='abs_logerror')
    plt.title(f"Mean Absolute Log Error by Cluster for cluster: {cluster_name}")
    plt.ylabel("Mean Absolute Log Error", fontsize=16)
    plt.xlabel("Cluster", fontsize=16)
    
    # Run significance test (t-test)
    overall_mean = train_scaled.abs_logerror.mean()
    print(f"Overall mean logerror: {overall_mean}")
    for col in set(train_scaled[cluster_name]):
        sample = train_scaled[train_scaled[cluster_name]==col]
        t, p = stats.ttest_1samp(sample.abs_logerror, overall_mean)
        print(col, "Significant? ", p<alpha, "t value: ", t)
        
    return train_scaled

def plot_bar_comparison(train_scaled, feature_to_cluster, cluster_title, cluster_names):
    
    sns.barplot(data = train_scaled, x = cluster_name, y='abs_logerror')
    plt.title(f"Mean Absolute Log Error by Cluster for {cluster_title}")
    plt.ylabel("Mean Absolute Log Error", fontsize=16)
    plt.xlabel("Cluster", fontsize=16)
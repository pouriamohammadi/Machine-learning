import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

"""## Uploading Vermont Culverts Dataset"""
# Upload Vermont Culvert Dataset
culverts = pd.read_csv("Vermont_culverts.csv")
culverts.head(5)

"""## **Analysing All Features**"""

culverts.drop(['id', 'owner', 'town', 'latitude', 'longitude', 'drop_inlet', "road", 
               "rd_imp_ovr", "bankfull_w", "bankfull_w_source", "erosion", "hdr_cond", 
               "cover_dpt", "dir_output", "perched", "need_clean", "twn_hwy_cd", "local_id",
               "location", "x_coord", "y_coord", "historic", "label", "comment1", "comment2", 
               "comment3", "comment4", "comment_no"], axis=1, inplace=True)
# culverts = culverts.fillna(culverts.mode().iloc[0])
# culverts['cul_type'].unique()
# culverts.to_csv('Vermont_dropped.csv',na_rep='nan',header=True ,index=False,float_format=None)

#### Alignment (alignment)
# Is the culvert poorly aligned with the stream channel that it is crossing? "Yes" and "No"
# culverts['alignment'].unique()
# culverts.drop(culverts.index[culverts['alignment'] == 'Unknown'], inplace = True)
culverts.loc[(culverts.alignment == 'Unknown'),'alignment'] = np.nan
culverts['alignment'].fillna(value=culverts['alignment'].mode()[0], inplace=True)
tempdf = pd.get_dummies(culverts["alignment"], prefix='aligned')
culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
culverts.drop(["alignment"], axis=1, inplace=True)

#### Outlet_area (outlet_area)
culverts.dropna(subset=['height'], inplace=True)
culverts.drop(culverts.index[culverts['height'] > 100], inplace = True)
culverts.drop(culverts.index[culverts['height'] <= 1], inplace = True)
culverts["height"] = culverts["height"].astype(int)
# culverts['height'].unique()
# culverts.hist(column='height',bins=100)
culverts.drop(culverts.index[culverts['width'] <= 1], inplace = True)
culverts["outlet_area"] = culverts["height"] * culverts["width"]
culverts.drop(culverts.index[culverts['outlet_area'] > 1000], inplace = True)
# culverts.hist(column='outlet_area',bins=100)
def outlet_distribution(data):
    data.loc[data['outlet_area'] <= 200, 'out_size'] = 'small'
    data.loc[(data['outlet_area'] > 200) & (data['outlet_area'] <= 500), 'out_size'] = 'medium'
    data.loc[(data['outlet_area'] > 500), 'out_size'] = 'large'
    return data
# Categorizng outlet
culverts = outlet_distribution(culverts)
tempdf = pd.get_dummies(culverts["out_size"], prefix='size')
culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
culverts.drop(["out_size"], axis=1, inplace=True)
culverts.drop(["outlet_area"], axis=1, inplace=True)
culverts.drop(["width"], axis=1, inplace=True)
culverts.drop(["height"], axis=1, inplace=True)
culverts.drop(["length"], axis=1, inplace=True)

# culverts["width"] = culverts["width"].astype(int)
# culverts["length"] = culverts["length"].astype(int)
# culverts["outlet_area"] = culverts["outlet_area"].astype(int)

#### Type of structure (cul_type)
culverts = culverts[culverts.cul_type != "Unknown"]
# culverts.loc[(culverts.cul_type == "Unknown"),'cul_type'] = np.nan
# culverts['cul_type'].fillna(value=culverts['cul_type'].mode()[0], inplace=True)
tempdf = pd.get_dummies(culverts["cul_type"], prefix='type')
culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
culverts.drop(["cul_type"], axis=1, inplace=True)

#### Material of structure (cul_matl)
culverts = culverts[culverts.cul_matl != "Unknown"]
# culverts.loc[(culverts.cul_matl == "Unknown"),'cul_matl'] = np.nan
# culverts['cul_matl'].fillna(value=culverts['cul_matl'].mode()[0], inplace=True)
tempdf = pd.get_dummies(culverts["cul_matl"], prefix='matl')
culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
# culverts.drop(["cul_matl"], axis=1, inplace=True)

# culverts.to_csv('Vermont_dropped.csv',na_rep='nan',header=True ,index=False,float_format=None)

#### Header Material (hdr_matl)
culverts['hdr_matl'].unique()
# culverts.drop(culverts.index[culverts['hdr_matl'] == 'Unknown'], inplace = True)
# culverts.drop(culverts.index[culverts['hdr_matl'] == 'None'], inplace = True)
# culverts = culverts[culverts['hdr_matl'].notna()]
# culverts.loc[(culverts.hdr_matl == 'Unknown'),'hdr_matl'] = np.nan
culverts.loc[(culverts.hdr_matl == 'None'),'hdr_matl'] = "Unknown"
culverts['hdr_matl'].fillna(value= "Unknown", inplace=True)
# culverts['hdr_matl'].fillna(value=culverts['hdr_matl'].mode()[0], inplace=True)
def filling_unknown(data):
    df = data[data.hdr_matl == "Unknown"]
    materials_list = df['cul_matl'].unique()
    for m in materials_list :
        df_m = data[data.hdr_matl != "Unknown"]
        df_m = df_m.loc[culverts['cul_matl'] == m]
        header_material= df_m['hdr_matl'].mode()[0]
        # print (data['hdr_matl']=np.nan)
        data.loc[(data['cul_matl']==m) & (data['hdr_matl']== "Unknown"),'hdr_matl'] = header_material
        # data.loc[(data['cul_matl']==m) , ['hdr_matl']].fillna(value=culverts['hdr_matl'].mode()[0], inplace=True)
    return data
culverts = filling_unknown(culverts)
tempdf = pd.get_dummies(culverts["hdr_matl"], prefix='hdr_matl')
culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
culverts.drop(["hdr_matl"], axis=1, inplace=True)
culverts.drop(["cul_matl"], axis=1, inplace=True)

#### Road Importance (rd_imp) 
#(Scores range from 1-3, with higher scores indicating a higher priority)
# culverts.drop(culverts.index[culverts['rd_imp'] == 0], inplace = True)
# culverts = culverts[culverts['rd_imp'].notna()]
culverts.loc[(culverts.rd_imp == 0),'rd_imp'] = np.nan
culverts['rd_imp'].fillna(value=culverts['rd_imp'].mode()[0], inplace=True)
tempdf = pd.get_dummies(culverts["rd_imp"], prefix='rd_imp')
culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
culverts.drop(["rd_imp"], axis=1, inplace=True)

#### year_built (Year structure was built, if known)
culverts.drop(culverts.index[culverts['year_built'] == 0], inplace = True)
culverts.dropna(subset=['year_built'], inplace=True)
culverts["year_built"] = pd.to_numeric(culverts["year_built"])
culverts = culverts.drop(culverts[(culverts['year_built'] <= 1900) | (culverts['year_built'] >= 2022)].index)
culverts["year_built"] = culverts["year_built"].astype(int)
culverts=culverts.reset_index(drop=True)

#### inv_dt (Date of inspection/inventory)
culverts.dropna(subset=['inv_dt'], inplace=True)
culverts['date'] = pd.to_datetime(culverts['inv_dt'])
culverts['invetory_year'] = culverts['date'].dt.year
culverts.drop(["inv_dt", "date"], axis=1, inplace=True)
culverts["invetory_year"] = culverts["invetory_year"].astype(int)

#### Age
culverts['Age'] = culverts['invetory_year'] - culverts['year_built']
culverts = culverts[culverts.Age >= 0]
culverts["Age"] = culverts["Age"].astype(int)
def age_distribution(data):
    data.loc[data['Age'] <= 10, 'AgeGroup'] = 1
    data.loc[(data['Age'] > 10) & (data['Age'] <= 20), 'AgeGroup'] = 2
    data.loc[(data['Age'] > 20) & (data['Age'] <= 30), 'AgeGroup'] = 3
    data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'AgeGroup'] = 4
    data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'AgeGroup'] = 5
    data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'AgeGroup'] = 6
    data.loc[(data['Age'] > 60) & (data['Age'] <= 70), 'AgeGroup'] = 7
    data.loc[(data['Age'] > 70), 'AgeGroup'] = 8
    data['AgeGroup'].astype(int)
    return data
# Categorizing Age
# culverts = age_distribution(culverts)
# tempdf = pd.get_dummies(culverts["AgeGroup"], prefix='AgeGroup')
# culverts = pd.merge(left=culverts, right=tempdf, left_index=True, right_index=True)
# culverts.drop(['Age'], axis=1, inplace=True)
culverts.drop(['year_built'], axis=1, inplace=True)


#### oa_cond (The condition of the culvert)
# Excellent - recently constructed, no visible deficiencies;
# Good - at least 75% open, few if any minor deficiencies;
# Fair - at least 50% open, some existing or developing deficiencies;
# Poor - at least 25% open and/or has serious deficiencies;
# Critical - less than 25% open and/or has critical deficiencies;
# Urgent - Critical deficiencies have forced the structure to be closed. Structure is closed to traffic;
# Unknown - cannot provide a reasonable evaluation due to the structure not being visible, property owner, etc.

culverts = culverts[culverts.oa_cond != "Unknown"]
cleanup_nums = {"oa_cond": {"Closed": 0, "Urgent": 0, "Critical": 1, "Poor": 2,
                            "Fair": 3, "Good": 3, "Excellent": 4}}
culverts = culverts.replace(cleanup_nums)
culverts["oa_cond"] = culverts["oa_cond"].astype(int)

culverts_new = culverts.copy()

culverts_new.drop(['invetory_year'], axis=1, inplace=True)

#split data into train and test
Y = culverts_new['oa_cond']
X = culverts_new.drop(['oa_cond'],axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
print ("preprocess done")

# Choosing classifier
print("Please Choose Classifier SVM=1 , Randomforest=2 ??")
classifier = int(input())

def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        accuracy = metrics.accuracy_score(test_labels,predictions)
        print('Model Performance')
        # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))
        return accuracy

"""## Random Forest Classifier with tuning hyperparameters"""
if classifier == 2:
    print("SVM chosen for classifier")
    # Number of trees in random forest
    n_estimators = np.arange(100, 2000, step=10)
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = list(np.arange(10, 100, step=10)) + [None]
    # Minimum number of samples required to split a node
    min_samples_split = np.arange(2, 10, step=2)
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                                    cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, Y_train)
    # params = rf_random.best_params_
    
    #base random forest
    base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
    base_model.fit(X_train, Y_train)
    base_accuracy = evaluate(base_model, X_test, Y_test)
    #best random forest
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, Y_test)
    print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
    
    # Features Importance
    # feature_list = list(X.columns)
    # feature_imp = pd.Series(best_random.feature_importances_, index=feature_list).sort_values(ascending=False)
    # print(feature_imp)
    
    # results of test set
    print("results of test set")
    grid_predictions_test = best_random.predict(X_test)
    print(classification_report(Y_test, grid_predictions_test))
    # results of train set
    print("results of train set")
    grid_predictions_train = best_random.predict(X_train)
    print(classification_report(Y_train, grid_predictions_train))
    #cross validation
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores = cross_val_score(best_random, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy based of cross validation: %.3f (%.3f)' % (mean(scores), std(scores)))
    
"""## SVC with tuning hyperparameters"""
if classifier == 1:
    # Create the random grid
    print("SVM chosen for classifier")
    param_grid = { 'C':[0.1,1,10,100,1000],
                  'kernel':['rbf'],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    # SVC_classifier = SVC(C=10, kernel='rbf')
    # SVC_classifier.fit(X_train, Y_train)
    # best_SVC = SVC_classifier
    # scores = cross_val_score(SVC_classifier, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    
    # SVC_classifier.fit(X_train, Y_train)
    # SVC_classifier = SVC()
    # SVC_random = RandomizedSearchCV(estimator = SVC_classifier, param_distributions = param_grid, n_iter = 100,
                                    # cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # SVC_random.fit(X_train, Y_train)
    # best_SVC = SVC_random.best_estimator_
    # random_accuracy = evaluate(best_SVC, X_test, Y_test)
    
    SVC_grid = GridSearchCV(SVC(),param_grid)
    SVC_grid.fit(X_train,Y_train)
    best_SVC = SVC_grid.best_estimator_
    # results of test set
    print("results of test set")
    grid_prediction_test = best_SVC.predict(X_test)
    print(classification_report(Y_test, grid_prediction_test))
    # results of train set
    print("results of train set")
    grid_predictions_train = best_SVC.predict(X_train)
    print(classification_report(Y_train, grid_predictions_train))
    #cross validation
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores = cross_val_score(best_SVC, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy based of cross validation: %.3f (%.3f)' % (mean(scores), std(scores)))


    

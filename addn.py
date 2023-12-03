from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE
import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

# Performing Standard Scaler on X train and test
from sklearn.preprocessing import StandardScaler

def standard_scaler():
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X.columns)
    return X_train_scaled,X_test_scaled

# Importing dataset
df = pd.read_csv('../Data/heart_failure_clinical_records_dataset orig.csv')

y = df['DEATH_EVENT'] 
X = df.drop('DEATH_EVENT',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


smote_up = SMOTE(random_state = 42)
X_train , y_train = smote_up.fit_resample(X_train,y_train)

print(f'X_train order : {X_train.shape}')
print(f'y_train order : {y_train.shape}')
print(f'X_test order : {X_test.shape}')
print(f'y_test order :{y_test.shape}')

print("Class distribution : ",Counter(y_train))

def top_feat(model,feat_cnt):
    '''
    This function provides the top "n" features for provided model using RFE.
    '''
    rfe = RFE(model, n_features_to_select=feat_cnt)
    rfe.fit(X_train, y_train)
    return X.columns[rfe.support_].to_list()



Acc_list = []

def hf_model(model,model_name,features=""):
    # Feature Scaling
    if model_name in ['GaussianNB','XGboost','RandomForest','GradientBoosting',
                      'LGBMClassifier','CatBoostClassifier','DecisionTree']:
        X_train_temp,X_test_temp = X_train,X_test
    else:
        X_train_temp,X_test_temp = standard_scaler()
    
    # Feature Selection
    if model_name not in ['KNN','GaussianNB','SVM']:
        X_train_temp = X_train_temp[features]
        X_test_temp = X_test_temp[features]
   
    # Fitting train set
    model.fit(X_train_temp,y_train)

    # Performing prediction on test set
    y_test_pred = model.predict(X_test_temp)
        
    # Calculating Evaluation Metrics
    s1 = accuracy_score(y_test,y_test_pred)
    print(f"{model_name} Success Rate :{round(s1*100,2)}%\n")
    print(classification_report(y_test,y_test_pred))

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test_temp)[:, 1]
    else:
        # For models without predict_proba, use decision function for SVM
        probabilities = model.decision_function(X_test_temp)


    fpr, tpr, _ = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)
    
    with open('cat_model.pkl', 'wb') as model_file:
        pkl.dump(cat, model_file)

    print(y_test_pred[0])

    return {
        'model_name': model_name,
        'prediction': int(y_test_pred[0]),  
        'probability': f"{round(max(probabilities) * 100, 2)}%",  
    }


cat = CatBoostClassifier(learning_rate=0.01, n_estimators=1000,verbose=0)
features = top_feat(cat,10)
hf_model(cat,"CatBoostClassifier",features)
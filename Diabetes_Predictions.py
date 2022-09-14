
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df=pd.read_csv('diabetes.csv')

def preprocess(data):
    print('Preprocessing the data')
    data['Glucose']=np.where(data['Glucose']<10,data['Glucose']==10,data['Glucose']) 
    data['Insulin']=np.where(data['Insulin']==0,data['Insulin'].median(),data['Insulin']) 
    data['SkinThickness']=np.where(data['SkinThickness']<10,data['SkinThickness']==10,data['SkinThickness']) 
    data['BMI']=np.where(data['BMI']<10,data['SkinThickness'].median(),data['SkinThickness']) 
    data['BloodPressure']=np.where(data['BloodPressure']<20,data['BloodPressure']==20,data['BloodPressure']) 
    data['Pregnancies']=np.where(data['Pregnancies']<0,data['Pregnancies']==0,data['Pregnancies']) 
    return data
  
 
def remove_outliers(data):
    print('Detecting and removing the outliers')
    for feature in data.columns:
        quantile1,quantile3=np.percentile(data[feature],[25,75])     # Defining the quantile ranges

        iqr=quantile3-quantile1                                    # Defining inter quantile range      
        lower_bound=quantile1-(1.5*iqr)                            # Defining lower bound of outliers
        upper_bound=quantile3+(1.5*iqr)                            # Defining upper bound of outliers
        data=data[(data[feature]<=upper_bound)&(data[feature]>=lower_bound)]   # Taking only that data which is under the boundry we defined
        return data



def splitting(data):
    print('Splitting the data')
    X_train,X_test,y_train,y_test=train_test_split(data.drop('Outcome',axis=1),data['Outcome'],test_size=0.2,random_state=0)
    return X_train, X_test, y_train, y_test

def scaling(training, testing):
    print('Scaling the data')
    scaler=StandardScaler()
    training=scaler.fit_transform(training)
    testing=scaler.transform(testing)
    print('Saving a Scaler to "scaler" folder')
    with open('scaler/scaling.save','wb') as f_out:
        joblib.dump(scaler, f_out)
        
    return training, testing


def training_model(X_train, X_test, y_train, y_test):
    print('Training the SVM model')
    model=SVC()

    params={'C'         : [1,5,10,20,25,40,50,75,80,100],
            'kernel'    : ['linear','rbf'],
            'gamma'     : [1e-10,1e-8,1e-5,1e-2,1e-1,1,5,10,20,30,50,70,90,100]}
    
    grid_search=GridSearchCV(model,param_grid=params,scoring='roc_auc',cv=10,n_jobs=-1).fit(X_train,y_train)
    print(f'SVM model is trained and best parameters are {grid_search.best_params_}')
    print('Saving a model to "Models" folder')
    with open('models/svm.pkl', 'wb') as f_out:
        pickle.dump(grid_search.best_estimator_,f_out)
    return grid_search.best_estimator_

def evaluation(X_test, y_test):
    model = training_model(X_train, X_test, y_train, y_test)
    print('Evaluating the model')

    y_pred=model.predict(X_test)

    score=accuracy_score(y_test,y_pred)
    confusion=confusion_matrix(y_test,y_pred)
    report=classification_report(y_test,y_pred)
    print(' Model: Support Vector Machine\n Accuracy of model: {:.2f}'.format(score))
    print(' Confusion Matrix: \n ',confusion)
    print('Classification report: \n',report)
    return score

if __name__ == '__main__':
    df = preprocess(df)
    df = remove_outliers(df)

    X_train, X_test, y_train, y_test = splitting(df)
    X_train, X_test = scaling(X_train, X_test)
    score = evaluation(X_test, y_test)


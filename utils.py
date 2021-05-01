'''
Utility functions for the churn rate prediction project
'''
import pandas as pd
import numpy as np

from sklearn.utils import resample
import datetime as dt

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (confusion_matrix ,classification_report, roc_curve, precision_recall_curve, auc
                                , make_scorer, recall_score, accuracy_score
                                , precision_score)

import matplotlib.pyplot as plt

def get_df_stats(df):
    '''
    Util function: return statistics of the input dataframe
    Params:
    df: pd.DataFrame
    Return:
    pd.DataFrame 
    '''
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0],df[col].isnull().sum(), df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_testues', 'Percentage of missing values', 'Number of missing values', 'Percentage of values in the biggest category', 'dtype'])
    return stats_df.sort_values('Percentage of missing values', ascending=False)





#create feature dataframe
def num2str(df):
    '''
    Replace str labels {1.0,0.0} -> {'yes','no'}
    '''  
    return df.replace({1.0:'yes',0.0:'no'})

def str2num(df):
    '''
    Replace str labels {'yes','no'} -> {1,0}
    '''  
    return df.replace({'yes':1.0,'no':0.0})
    
class DataProcessing( BaseEstimator, TransformerMixin):
    '''
    Data processing class. Recives raw dataset as an input returns featue model
    to use in the ML component
    '''
    def __init__( self, 
                        numfeatures=None 
                        ,catfeatures=None 
                        ,vars_to_drop =None 
                        ,to_log1p=True
                        ,columns_names = None
                        ):
        
        self.columns_names = columns_names
        self.numfeatures = numfeatures
        self.catfeatures = catfeatures
        self.vars_to_drop = vars_to_drop 
        
        self.to_log1p = to_log1p
    
    def resample_up(self,df,class_names={'majority':'no','minority':'yes'},seed=42):
        
        # Separate majority and minority classes
        df_majority = df[df.churn==class_names['majority']]
        df_minority = df[df.churn==class_names['minority']]
        
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,     # sample with replacement
                                        n_samples=df_majority.shape[0],    # to match majority class
                                        random_state=seed) # reproducible results

        return pd.concat([df_majority,df_minority_upsampled])

    def fit( self, X, y = None ):
        '''
        Placeholder for fitting the model
        '''
        return self 
    
    def transform( self, X, y = None ):
        '''
        Define features transformation 
        '''
        # print('===> Starting featues processing')
        #PROCESS FEATUES 
        _X = X.copy()
        
        #recompose dataframe if np array is proviced 
        if not isinstance(_X,pd.DataFrame):
            _X = pd.DataFrame(_X,columns=self.columns_names)

        #remove useless features 
        if self.vars_to_drop:
            _X = _X.drop(self.vars_to_drop,axis=1)

        #Transform categorical features
        _X[self.catfeatures] = str2num(_X[self.catfeatures])

        #transform to float
        _X = _X.astype(float)
        
        #fillna
        _X = _X.fillna(_X.dropna().mean())

        #Bining
        #handle account length feature
        _X['account_length_bin'] = _X['account_length'].map(lambda x: len(str(x)))
        del _X['account_length']



        
        
        #log1p transform
        if self.to_log1p:
            _X[self.numfeatures] = np.log1p(_X[self.numfeatures])
        
        # print('===> Features processing finished; X shape:', _X.shape)
        self.features_names = _X.columns.tolist()

        return _X

'''
Define nessesery function

'''
class DataLayer:
    '''
    Data Layer class. It contain both training and val and testing datasets
    '''
    def __init__(self,X,y,X_oos=None,target_label=None,test_size=0.33,random_state=42):
        '''
        Method that splits input data

        Train-Val data:
        X, y : pd.DataFrame

        Out-of-sample prediction data:
        X_oos
        '''
        #PROCESS LABELS 
        y_num = str2num(y)      
        
        #PROCESS OUT-OF-SAMPLE DATA
        if X_oos:
            X_oos = X_oos.drop([target_label],axis=1)
        #Split train-val data
        X_train, X_test, y_train, y_test = train_test_split(X, y_num
                            ,test_size=test_size
                            ,random_state=random_state)
        
        #RESAMLE ONLY TRAINING DATA
        #apply resampling 
        Xy_train = DataProcessing().resample_up(pd.concat([X_train,y_train],axis=1),class_names={'majority': 0.0,'minority': 1.0 })
        X_train , y_train = Xy_train.drop([target_label],axis=1) , Xy_train[[target_label]]

        self.X_train, self.X_test, self.y_train, self.y_test , self.X_oos = X_train, X_test, y_train, y_test , X_oos


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]
    
def precision_recall_threshold(p, r, thresholds,y_scores, t=0.5,title_prefix=''):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(y_scores, t)
    # print(pd.DataFrame(confusion_matrix(dl.y_test, y_pred_adj),
    #                    columns=['pred_neg', 'pred_pos'], 
    #                    index=['neg', 'pos']))
    
    # plot the curve
    fig, ax = plt.subplots(1,figsize=(8,8))
    
    ax.step(r, p, color='b', alpha=0.2,
             where='post')
    ax.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
#     ax.ylim([0.5, 1.01]);
#     ax.xlim([0.5, 1.01]);
    ax.set(title=f"{title_prefix}: Precision and Recall curve. Threshold at {t}"
        ,xlabel='Recall', ylabel='Precision')
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    ax.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,title_prefix=''):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.set(title=f"{title_prefix}: Precision and Recall Scores as a function of the decision threshold"
                ,ylabel="Score",xlabel= "Decision Threshold")
    ax.legend(loc='best')

def plot_roc_curve(fpr, tpr, label=None,title_prefix=''):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    
    ax.set(title=f'{title_prefix}: ROC Curve',xlabel="False Positive Rate"
        ,ylabel="True Positive Rate (Recall)")
    ax.legend(loc='best')


def print_report(y,y_hat):
    '''
    print classification report and confision matrix
    '''
    # confusion matrix on the test data.
    print('Confusion matrix:')
    print(pd.DataFrame(confusion_matrix(y, y_hat),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos'])
                 
        ,'\n',classification_report(y,y_hat))


def plot_reports(y_test,y_scores,score_to_optimize,initial_threshold=0.5,title_prefix=''):
    '''
    With `y_test` as an actual data and `y_scores` as model predictions, create model performance report
    '''
    p, r, thresholds = precision_recall_curve(y_test, y_scores)

    precision_recall_threshold(p, r, thresholds,y_scores,initial_threshold,title_prefix+'1')
    plot_precision_recall_vs_threshold(p, r, thresholds,title_prefix+'2')

    fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)
    print(auc(fpr, tpr)) # AUC of ROC
    plot_roc_curve(fpr, tpr, score_to_optimize,title_prefix+'3')

def plot_feature_importance(importance,features_names,title):
    fig, ax = plt.subplots(figsize=(8,4))
    
    # summarize feature importance
    for name ,v in zip(features_names,importance):
        print(f'Feature: {name}, Score: {v}')
    # plot feature importance
    (pd.DataFrame(sorted(zip(features_names,importance),key=lambda x: x[1] ,reverse=True)
    ,columns=['name','score'])
    .set_index('name').plot.bar(ax=ax,label='feature importance')
    )
    ax.set(title=title)
    ax.legend()


def run_grid_search(dl,clf,param_grid,refit_score='precision_score',model_name=''):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    print('*'*60,'\nStarting training model:',model_name)
    start_ = dt.datetime.now()
    skf = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(clf, param_grid
                                ,scoring={
                                        'precision_score': make_scorer(precision_score),
                                        'recall_score': make_scorer(recall_score),
                                        'accuracy_score': make_scorer(accuracy_score)
                                            }
                                ,refit=refit_score
                                ,cv=skf
                                ,verbose = 0
                                ,return_train_score=True
                                , n_jobs=-1)
    grid_search.fit(dl.X_train.values, dl.y_train.values)

    # make the predictions
    y_hat = grid_search.predict(dl.X_test.values)

    print('Best params for {}'.format(refit_score)
        ,':',grid_search.best_params_
        ,'\nGrid Search best scores: ',grid_search.best_score_
        ,'\nTime took (sec.):',(dt.datetime.now() - start_).seconds
        )
    print_report(dl.y_test,y_hat)
    
    return grid_search.best_estimator_






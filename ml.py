#######################################
#ML
#http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html#reading-multivariate-analysis-data-into-python
#ensemble refs: https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
#
#usage python ml.py -input filename -mode training or optimization or validation or voting -n kmer
#######################################
#standard modules
########################

import scipy
import numpy as np
import pickle
import pandas as pd
import os,sys,argparse
from random import randint
import getopt
from scipy import interp
from sklearn.externals import joblib
import seaborn as sns
########################
#sklearn modules
########################
import sklearn.metrics as metrics
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.model_selection import StratifiedKFold
#from sklearn.multiclass import OneVsRestClassifier
#models
#from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
#import xgboost
#import xgboost as xgb
#from xgboost import XGBClassifier
#tuning parameters
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

#visualization
import matplotlib.pyplot as plt
#roc curve
from sklearn.metrics import roc_curve,auc

#to evaluate classification accuracy
from sklearn.metrics import confusion_matrix

#visualize Decision Tree
import graphviz 
from sklearn import tree
#interpret tree classifier
from treeinterpreter import treeinterpreter as ti

sns.set_style('darkgrid')
sns.set_palette('colorblind')
blue, green, red, purple, yellow, cyan = sns.color_palette('colorblind')

def splitting(filename,seed):
    #set seed
    seed=seed
    #read data
    data=pd.read_table(filename,header=0)
    data.label=data.label.astype(int)
    #balance positive and negatives
    p=data[data['label']==int(1)].reset_index(drop=True)
    n=data[data['label']==int(0)].reset_index(drop=True)
    print('positives:\n',p.shape[0],'negatives:\n',n.shape[0])
    size=min(p.shape[0],n.shape[0])
    data=pd.concat([p.sample(n=size),n.sample(n=size)],axis=0).reset_index(drop=True)
    print(data.head,'\n',data.shape[0])
    #get array
    X=data.loc[:, data.columns != 'label'].astype(float).values
    Y=data.label.values
    validation_size=0.2
    #split in train ans validation set
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    return X_train, X_validation, Y_train, Y_validation


def prediction(model,X_test):
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    auc=metrics.auc(fpr, tpr)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Auc:%.2f%" % auc)

def voting(X,Y,kfold,estimators):
    # create the ensemble model
    #estimators are models
    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
    print(results.mean())

def histogram(data, bins, fileName,n,k):
    n, bins, patches = plt.hist(data, bins, histtype='bar', stacked=True)
    plt.grid(True)
    plt.savefig('img/'+fileName+'_'+k+'.png')
    plt.close()

def print_confusion_matrix(y_true, y_pred):
    labels = [int(0),int(1)]
    print(y_pred)
    cm = confusion_matrix(y_true, y_pred,labels=labels)
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    print("true negatives:",tn,"\n","false positives: ",fp,"\n","false negatives: ",fn,"\n","true positives: ",tp)

    #0\n1
    #cm=pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print("Count\n", cm)
    #plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.gray_r)
    #print("Percentage\n",np.true_divide(cm,cm.sum(axis=1)))
    #cm.plot()
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(cm)
    #plt.title('Confusion matrix of the classifier')
    #fig.colorbar(cax)
    #ax.set_xticklabels([''] + labels)
    #ax.set_yticklabels([''] + labels)
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    #plt.show()
    #plt.close()


def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)



def collect_models(seed,n_estimators):
    seed=seed
    models=list()
	#models.append(('SVM', SVC(decision_function_shape='ovo')))
    #models.append(('SVM', SVC(class_weight='balanced', kernel= 'rbf', gamma= 1000, C=0.001,probability=True)))
    #models.append(('SVM', SVC(class_weight='balanced', kernel= 'rbf', gamma= 1000, C=0.001,probability=True)))
    models.append(('SVM', SVC(class_weight='balanced', kernel= 'poly', gamma= 10, C=1,probability=True,degree= 4)))

    #{'C': 1, 'gamma': 1000, 'class_weight': 'balanced', 'kernel': 'rbf'}
    #XGB=Xgboost(X_train,Y_train)
    #models.append(('XGB',XGB))
    #models.append(('LR', LogisticRegression()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors= 3, p= 3, algorithm= 'auto')))
    models.append(('DTC', DecisionTreeClassifier(min_samples_split=3, max_depth= depth, max_features= "log2")))
    #models.append(('NB', GaussianNB()))
    models.append(('RF',RandomForestClassifier(n_jobs=1,max_depth=depth, min_samples_split= 3,n_estimators=n_estimators, random_state=1,criterion='entropy',max_features="log2")))
    #{'max_features': 'log2', 'max_depth': None, 'n_estimators': 500, 'min_samples_split': 2}
    #models.append(('DTC',DecisionTreeClassifier(criterion='entropy',max_depth=depth)))
    models.append(('ExtraTC',ExtraTreesClassifier(n_estimators=n_estimators,max_depth=depth,min_samples_split=2, random_state=0,bootstrap=True,n_jobs=1,max_features="sqrt")))
    models.append(('ABoostC',AdaBoostClassifier(n_estimators=n_estimators,learning_rate=0.1,random_state=1,algorithm='SAMME')))
    models.append(('GBoost',GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed)))
    return models
def training(X,Y,seed,scoring,k,n_estimators):

    names = list()
    means=list()
    models=collect_models(seed,n_estimators)
    for name, model in models:
        skf = StratifiedKFold(n_splits=10)
        cv=skf.get_n_splits(X, Y)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        for i,(train_index, test_index) in enumerate(skf.split(X,Y)):
            results = list()
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            #for name, model in models:
            m=model.fit(X_train,y_train)
            filename='{}.{}.pkl'.format(name,str(i))
            save_model(m,name,k)
            probas_=m.predict_proba(X_test)
            y_pred=m.predict(X_test)
            print(name,'\n',i,'\n')
            print(y_test,'\t',"y pred:",y_pred,'\n')
            print_confusion_matrix(y_test,y_pred)
            #print(probas_)
            #plot_ROC(X_train,y_train,X_test,y_test,model,n_classes=2)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            #predictions = model.predict_proba(X_test) > thresh
            #print(fpr,tpr)
            roc_auc= auc(fpr, tpr)
            #print(roc_auc)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6,0.6,0.6), label='Luck')
            
        mean_tpr /= cv
        #print(mean_tpr)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic '+name)
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig('img/'+name+'_'+k+'.png')
        plt.close()
        means.append([mean_fpr, mean_tpr,mean_auc])
        names.append(name)
    d=dict(zip(names,means))

        #print("%s: %f" % (name, np.mean(results)))
    #print(means,names)
    for j,name in enumerate(names):
        plt.plot(means[j][0], means[j][1],lw=1, label='{}_{}-auc:{}'.format('ROC fold',name,str(round(means[j][2],2))))
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        #plt.show()
    plt.savefig('img/'+k+'.roc_all_models.png')
    plt.close()
    return models

def save_model(model,filename,k):
    pickle.dump(model, open('models/'+filename+'_'+k, 'wb'))
    
def load_model(filename,X_validation,Y_validation):
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_validation, Y_validation)
    probas_=m.predict_proba(X_validation)
    y_pred=m.predict(X_validation)
    print(name,'\n',i,'\n')
    print_confusion_matrix(Y_validation,y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:,1])
    auc = auc(fpr,tpr)
    print('auc validation:',auc)
    print('accuarcy validation:',result)
    return result,auc,fpr,tpr

def tuning_SVM(X_train,Y_train,kfold,scoring):
    SVM=SVC()
    param_grid_SVM = {"kernel":['poly','linear','rbf','sigmoid'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'class_weight':['balanced', None],'degree':[3,4,5]}
    grid_search = GridSearchCV(estimator=SVM, param_grid=param_grid_SVM, cv=kfold, n_jobs=1, scoring=scoring)
    print(X_train,Y_train)
    grid_search.fit(X_train,Y_train)
    print('SVM\n',grid_search.best_params_)
    return grid_search.best_params_

def tuning_KNN(X_train,Y_train,kfold,scoring):
	KNN=KNeighborsClassifier()
	param_grid_KNN = {"algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],"n_neighbors":[2,3,4,5],"p":[1,2,3]}
	grid_search = GridSearchCV(estimator=KNN, param_grid=param_grid_KNN, cv=kfold, n_jobs=1, scoring=scoring)
	grid_search.fit(X_train,Y_train)
	print('KNN\n',grid_search.best_params_)
	return grid_search.best_params_

def tuning_tree(X_train,Y_train,kfold,scoring):
	DTC=DecisionTreeClassifier(random_state=1)
	param_grid_CART = {'max_features': ['sqrt','log2','auto',None],"min_samples_split":[2,3,4,5],"max_depth": [None,10,100,200]}
	grid_search = GridSearchCV(estimator=DTC, param_grid=param_grid_CART, cv=kfold, n_jobs=1, scoring=scoring)
	grid_search.fit(X_train,Y_train)
	print('DTC\n',grid_search.best_params_)
	return grid_search.best_params_

def tuning_RF(X_train,Y_train,kfold,scoring):
    RF=RandomForestClassifier(n_jobs=1, random_state=1,criterion='entropy')
    param_grid_RF = {'max_features': ['sqrt','log2','auto',None],"min_samples_split":[2,3,4,5],"n_estimators":[50,100,200,500],"max_depth": [None,10,100,200]}
    grid_search = GridSearchCV(estimator=RF, param_grid=param_grid_RF, cv=kfold, n_jobs=1, scoring=scoring)
    grid_search.fit(X_train,Y_train)
    print('RF\n',grid_search.best_params_)
    return grid_search.best_params_
'''
def tuning_XGB(X_train,Y_train,kfold):
    # grid search
    model = XGBClassifier()
    n_estimators = [100, 200, 300, 400, 500]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train,Y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #	print("%f (%f) with: %r" % (mean, stdev, param))
    # plot results
    scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
    for i, value in enumerate(learning_rate):
        pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    pyplot.legend()
    pyplot.xlabel('n_estimators')
    pyplot.ylabel('Log Loss')
    os.system('mkdir -p plot')
    pyplot.savefig('plot/n_estimators_vs_learning_rate.png')
'''
def graph(filename,X_train,y_train,X_test,y_test):
    # Set seaborn colours

    #ref https://github.com/gregtam/interpreting-decision-trees-and-random-forests/blob/master/Blog%20Code.ipynb
    rf=RandomForestClassifier(n_jobs=1,max_depth=depth, min_samples_split= 3,n_estimators=n_estimators, random_state=1,criterion='entropy',max_features="log2")
    rf=rf.fit(X_train,y_train)
    prediction, bias, contributions = ti.predict(rf,X_test)
    data=pd.read_table(filename,header=0)
    #print(data.head(),X_train,data.columns.str.replace("|","**").values)
    #Extra_clf=ExtraTreesClassifier(max_depth=depth,min_samples_split=2, random_state=0,max_features="sqrt")
    #print("y train",y_train)
    clf=tree.ExtraTreeClassifier(max_depth=depth,min_samples_split=2, random_state=0,max_features="sqrt")
    clf.fit(X_train,y_train)
    importances = clf.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    #print(indices,y_train)
    # Print the feature ranking
    print("Feature ranking:")
    
    #data.columns[:-1].str.replace("|","*")
    for f in range(X_train.shape[1]):
        print("{}. feature {} {} {:.3f}".format(f + 1, indices[f],data.columns[:-1].str.replace("|","*")[indices[f]],importances[indices[f]]))
    dot_data = tree.export_graphviz(clf, out_file=None, 
                        feature_names=data.columns[:-1].str.replace("|","*").values.astype('str'),  
                        class_names=y_train.astype(str), 
                        filled=True, rounded=True,  
                        special_characters=False) 
    graph = graphviz.Source(dot_data) 
    graph.render("img/Extratree_"+str(k)) 
    #dt_bin_clf_pred, dt_bin_clf_bias, dt_bin_clf_contrib = ti.predict(clf, X_test)
    rf_bin_clf_pred, rf_bin_clf_bias, rf_bin_clf_contrib = ti.predict(rf, X_test)
    print("Prediction", prediction)
    print("Bias (trainset prior)", bias)
    print("Feature contributions:",contributions)
    #for c, feature in zip(contributions[0],data.columns[:-1].str.replace("|","**").values.astype('str')):
        #print(feature, c,flush=True)
    #plot
    #print(rf,X_test,rf_bin_clf_contrib)


    X=data.loc[:, data.columns != 'label'].astype(float)
    Y=data.label
    df, true_label, scores = plot_obs_feature_contrib(rf,
                                                    rf_bin_clf_contrib.round(decimals=2),
                                                    X.iloc[0:10],
                                                    Y.iloc[0:10],
                                                    2,
                                                    num_features=10,
                                                    order_by='contribution',
                                                    violin=True
                                                )
    true_value_list = ["Cyto","Nucleus"]
    scores=[round(i,2) for i in scores ]
    score_dict = zip(true_value_list, scores)
    print("true label:",true_label)
    title = 'Contributions Class\nTrue Value: {}\nScores: {}'.format(true_value_list[true_label],
                                                ', '.join(['{} - {}'.format(i, j) for i, j in score_dict]))
    plt.title(title)
    plt.tight_layout()
    plt.savefig('img/contribution_plot_violin_bin_clf_rf_'+str(k)+'.png')



def plot_obs_feature_contrib(clf, contributions, features_df, labels, index, 
                             class_index=0, num_features=None,
                             order_by='natural', violin=False, **kwargs):
    """Plots a single observation's feature contributions.
    Inputs:
    clf - A Decision Tree or Random Forest classifier object
    contributions - The contributions from treeinterpreter
    features_df - A Pandas DataFrame with the features
    labels - A Pandas Series of the labels
    index - An integer representing which observation we would like to
            look at
    class_index - The index of which class to look at (Default: 0)
    num_features - The number of features we wish to plot. If None, then
                   plot all features (Default: None)
    order_by - What to order the contributions by. The default ordering
               is the natural one, which takes the original feature
               ordering. (Options: 'natural', 'contribution')
    violin - Whether to plot violin plots (Default: False)
    Returns:
    obs_contrib_df - A Pandas DataFrame that includes the feature values
                     and their contributions
    """
    def _extract_contrib_array():
        # If regression tree
        if len(contributions.shape) == 2:
            if class_index > 0:
                raise ValueError('class_index cannot be positive for regression.')
            contrib_array = contributions[index]
        # If classification tree
        elif len(contributions.shape) == 3:
            if class_index >= contributions.shape[2]:
                raise ValueError('class_index exceeds number of classes.')
            contrib_array = contributions[index, :, class_index]
        else:
            raise ValueError('contributions is not the right shape.')

        return contrib_array

    def _plot_contrib():
        """Plot contributions for a given observation. Also plot violin
        plots for all other observations if specified.
        """
        if violin:
            # Get contributions for the class
            if len(contributions.shape) == 2:
                contrib = contributions
            elif len(contributions.shape) == 3:
                contrib = contributions[:, :, class_index]

            contrib_df = pd.DataFrame(contrib, columns=features_df.columns)

            if has_ax:
                ax.violinplot([contrib_df[w] for w in obs_contrib_tail.index],
                              vert=False,
                              positions=np.arange(len(obs_contrib_tail))
                             )
                ax.scatter(obs_contrib_tail.contrib,
                           np.arange(obs_contrib_tail.shape[0]),
                           color=red,
                           s=100
                          )
                ax.set_yticks(np.arange(obs_contrib_tail.shape[0]))
                ax.set_yticklabels(obs_contrib_tail.index)

            else:
                #print(obs_contrib_tail.contrib,
                            #np.arange(obs_contrib_tail.shape[0]))
                # Plot a violin plot using only variables in obs_contrib_tail
                plt.violinplot([contrib_df[w] for w in obs_contrib_tail.index],
                               vert=False,
                               positions=np.arange(len(obs_contrib_tail))
                              )
                plt.scatter(obs_contrib_tail.contrib,
                            np.arange(obs_contrib_tail.shape[0]),
                            color=red,
                            s=100
                           )
                plt.yticks(np.arange(obs_contrib_tail.shape[0]),
                           obs_contrib_tail.index
                          )
        else:
            obs_contrib_tail['contrib'].plot(kind='barh', ax=ax)

        if has_ax:
            ax.axvline(0, c='black', linestyle='--', linewidth=2)
        else:
            plt.axvline(0, c='black', linestyle='--', linewidth=2)

        x_coord = ax.get_xlim()[0]
        obs_contrib_tail['feat_val']=[float('{:1.3f}'.format(i)) for i in obs_contrib_tail['feat_val']]
        for y_coord, feat_val in enumerate(obs_contrib_tail['feat_val']):
            if has_ax:
                t = ax.text(x_coord, y_coord, feat_val)
            else:
                t = plt.text(x_coord, y_coord, feat_val)
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=blue))

    def _edit_axes():
        if has_ax:
            ax.set_xlabel('Contribution of feature')
        else:
            plt.xlabel('Contribution of feature')

        true_label = labels.iloc[index]
        if isinstance(clf, DecisionTreeClassifier)\
                or isinstance(clf, RandomForestClassifier):
            scores = (clf.predict_proba(features_df.iloc[index:index+1])[0]).round(decimals=2)
            scores = [float('{:1.3f}'.format(i)) for i in scores]

            if has_ax:
                ax.set_title('True Value: {}\nScores: {}'
                                 .format(true_label, scores[class_index]))
            else:
                plt.title('True Value: {}\nScores: {}'
                              .format(true_label, scores[class_index]))

            # Returns obs_contrib_df (flipped back), true labels, and scores 
            return obs_contrib_df.iloc[::1], true_label, scores

        elif isinstance(clf, DecisionTreeRegressor)\
                or isinstance(clf, RandomForestRegressor):
            pred = clf.predict(features_df.iloc[index:index+1])[0]

            if has_ax:
                ax.set_title('True Value: {}\nPredicted Value: {:1.3f}'
                                 .format(true_label, pred))
            else:
                plt.title('True Value: {}\nPredicted Value: {:1.3f}'
                              .format(true_label, pred))

            # Returns obs_contrib_df (flipped back), true labels, and scores 
            return obs_contrib_df.iloc[::-1], true_label, pred

    if 'ax' in kwargs:
        has_ax = True
        ax = kwargs['ax']
    else:
        has_ax = False
        fig, ax = plt.subplots()

    feature_array = features_df.iloc[index]
    contrib_array = _extract_contrib_array()

    obs_contrib_df = pd.DataFrame({'feat_val': feature_array,
                                   'contrib': contrib_array
                                  })
    # Flip DataFrame vertically to plot in same order
    obs_contrib_df = obs_contrib_df.iloc[::-1]

    obs_contrib_df['abs_contrib'] = np.abs(obs_contrib_df['contrib'])
    if order_by == 'contribution':
        obs_contrib_df.sort_values('abs_contrib', inplace=True)

    # Trim the contributions if num_features is specified
    if num_features is not None:
        obs_contrib_tail = obs_contrib_df.tail(num_features).copy()
    else:
        obs_contrib_tail = obs_contrib_df.copy()

    _plot_contrib()
    return _edit_axes()


def plot_single_feat_contrib(feat_name, contributions, features_df,
                             class_index=0, class_name='', add_smooth=False,
                             frac=2/3, **kwargs):
    """Plots a single feature's values across all observations against
    their corresponding contributions.
    Inputs:
    feat_name - The name of the feature
    contributions - The contributions from treeinterpreter
    features_df - A Pandas DataFrame with the features
    class_index - The index of the class to plot (Default: 0)
    class_name - The name of the class being plotted (Default: '')
    add_smooth - Add a lowess smoothing trend line (Default: False)
    frac - The fraction of data used when estimating each y-value
           (Default: 2/3)
    """


    # Create a DataFrame to plot the contributions
    def _get_plot_df():
        """Gets the feature values and their contributions."""

        if len(contributions.shape) == 2:
            contrib_array = contributions[:, feat_index]
        elif len(contributions.shape) == 3:
            contrib_array = contributions[:, feat_index, class_index]
        else:
            raise Exception('contributions is not the right shape.')

        plot_df = pd.DataFrame({'feat_value': features_df[feat_name].tolist(),
                                'contrib': contrib_array
                               })
        return plot_df

    def _get_title():
        # Set title according to class_
        if class_name == '':
            return 'Contribution of {}'.format(feat_name)
        else:
            return 'Conribution of {} ({})'.format(feat_name, class_name)

    def _plot_contrib():
        # If a matplotlib ax is specified in the kwargs, then set ax to it
        # so we can overlay multiple plots together.
        if 'ax' in kwargs:
            ax = kwargs['ax']
            # If size is not specified, set to default matplotlib size
            if 's' not in kwargs:
                kwargs['s'] = 40
            plot_df\
                .sort_values('feat_value')\
                .plot(x='feat_value', y='contrib', kind='scatter', **kwargs)
            ax.axhline(0, c='black', linestyle='--', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel(feat_name)
            ax.set_ylabel('Contribution')
        else:
            plt.scatter(plot_df.feat_value, plot_df.contrib, **kwargs)
            plt.axhline(0, c='black', linestyle='--', linewidth=2)
            plt.title(title)
            plt.xlabel(feat_name)
            plt.ylabel('Contribution')

    def _plot_smooth():
        # Gets lowess fit points
        x_l, y_l = lowess(plot_df.contrib, plot_df.feat_value, frac=frac).T
        # Overlays lowess curve onto data
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.plot(x_l, y_l, c='black')
        else:
            plt.plot(x_l, y_l, c='black')

    # Get the index of the feature
    feat_index = features_df.columns.get_loc(feat_name)
    # Gets the DataFrame to plot
    plot_df = _get_plot_df()
    title = _get_title()
    _plot_contrib()

    if add_smooth:
        _plot_smooth()

if __name__=='__main__':
    #create directories
    if not os.path.exists('img'):
        os.makedirs('img')
    if not os.path.exists('models'):
        os.makedirs('models')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-input') # inputfile path
    parser.add_argument('-output') # outputfile label
    parser.add_argument('-kfold',default=10) # kfold default=10
    parser.add_argument('-mode') # training, validation, voting (keep the best models), optimization
    parser.add_argument('-n') # training, validation, voting (keep the best models), optimization

    args = parser.parse_args()
    depth=None
    n_estimators=100
    seed=7
    kfold=args.kfold
    scoring='roc_auc'
    filename=args.input
    mode=args.mode
    k=args.n
    #filename,mode=func(sys.argv[1:])
    X_train, X_validation, Y_train, Y_validation=splitting(filename,seed)
    #run_xgboost(X_train,Y_train)
    if mode=='training':
        model=training(X_train,Y_train,seed,scoring,k,n_estimators)
    
    if mode=='graph':
        graph(filename,X_train,Y_train,X_validation,Y_validation)
              
    if mode=='validation':
        filename_model=input("insert model path:\n")
        load_model(filename_model,X_validation,Y_validation)
    if mode=='voting':
        models=collect_models(seed)
        print(models,len(models))
        #voting
        voting(X_train,Y_train,kfold,models[3:5])
        estimators=list()
        for i in [0,3,4]:
            estimators.append(models[i])
        voting(X_train,Y_train,kfold,estimators)
        estimators=list()
        for i in [0,3,4,6]:
            estimators.append(models[i])
        voting(X_train,Y_train,kfold,estimators)
        estimators=list()
        for i in [0,1,3,4,6]:
            estimators.append(models[i])            
        voting(X_train,Y_train,kfold,estimators)
    if mode=='optimization':
        tuning_SVM(X_train,Y_train,kfold,scoring)
        #tuning_KNN(X_train,Y_train,kfold,scoring)
        #tuning_tree(X_train,Y_train,kfold,scoring)
        #tuning_RF(X_train,Y_train,kfold,scoring)
        #tuning_XGB(X_train,Y_train,kfold)

#https://www.kaggle.com/gbxiao/pytanic
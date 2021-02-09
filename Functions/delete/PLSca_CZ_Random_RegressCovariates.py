# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import numpy as np
import time
import random
from sklearn import linear_model
from sklearn import preprocessing
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSCanonical
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import statsmodels.formula.api as sm
from sklearn.decomposition import PCA
CodesPath = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho/scripts/AnalysisScripts/Functions';

def PLSca_APredictB_Bootstrap(Brain_Matrix_train, Behavior_Matrix_train, Covariates_train, Brain_Matrix_test, Behavior_Matrix_test, Covariates_test, Components_Number, BootStrapTimes, ResultantFolder):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder);
    Brain_Matrix_train_Mat = {'Brain_Matrix_train': Brain_Matrix_train}
    Brain_Matrix_train_Mat_Path = ResultantFolder + '/Brain_Matrix_train.mat'
    sio.savemat(Brain_Matrix_train_Mat_Path, Brain_Matrix_train_Mat);
    Behavior_Matrix_train_Mat = {'Behavior_Matrix_train': Behavior_Matrix_train}
    Behavior_Matrix_train_Mat_Path = ResultantFolder + '/Behavior_Matrix_train.mat'
    sio.savemat(Behavior_Matrix_train_Mat_Path, Behavior_Matrix_train_Mat);

    Brain_Matrix_test_Mat = {'Brain_Matrix_test': Brain_Matrix_test}
    Brain_Matrix_test_Mat_Path = ResultantFolder + '/Brain_Matrix_test.mat'
    sio.savemat(Brain_Matrix_test_Mat_Path, Brain_Matrix_test_Mat);
    Behavior_Matrix_test_Mat = {'Behavior_Matrix_test': Behavior_Matrix_test}
    Behavior_Matrix_test_Mat_Path = ResultantFolder + '/Behavior_Matrix_test.mat'
    sio.savemat(Behavior_Matrix_test_Mat_Path, Behavior_Matrix_test_Mat);

    for i in np.arange(BootStrapTimes):
        ResultantFolder_TimeI = ResultantFolder + '/Time_' + str(i)
        if not os.path.exists(ResultantFolder_TimeI):
            os.makedirs(ResultantFolder_TimeI);
        if not os.path.exists(ResultantFolder_TimeI + '/PLS_Prediction.mat'):
            Configuration_Mat = {'Brain_Matrix_train_Mat_Path': Brain_Matrix_train_Mat_Path, \
                                 'Behavior_Matrix_train_Mat_Path': Behavior_Matrix_train_Mat_Path, \
                                 'Covariates_train': Covariates_train, \
                                 'Brain_Matrix_test_Mat_Path': Brain_Matrix_test_Mat_Path, \
                                 'Behavior_Matrix_test_Mat_Path': Behavior_Matrix_test_Mat_Path, \
                                 'Covariates_test': Covariates_test, \
                                 'Components_Number': Components_Number, \
                                 'ResultantFolder_TimeI': ResultantFolder_TimeI};
            sio.savemat(ResultantFolder_TimeI + '/Configuration.mat', Configuration_Mat);

            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + CodesPath + '");\
                from PLSca_CZ_Random_RegressCovariates import PLSca_APredictB_Bootstrap_Sub;\
                import os;\
                import scipy.io as sio;\
                Configuration = sio.loadmat("' + ResultantFolder_TimeI + '/Configuration.mat");\
                Brain_Matrix_train_Mat_Path = Configuration["Brain_Matrix_train_Mat_Path"];\
                Behavior_Matrix_train_Mat_Path = Configuration["Behavior_Matrix_train_Mat_Path"];\
                Covariates_train = Configuration["Covariates_train"];\
                Brain_Matrix_test_Mat_Path = Configuration["Brain_Matrix_test_Mat_Path"];\
                Behavior_Matrix_test_Mat_Path = Configuration["Behavior_Matrix_test_Mat_Path"];\
                Covariates_test = Configuration["Covariates_test"];\
                Components_Number = Configuration["Components_Number"];\
                ResultantFolder_TimeI = Configuration["ResultantFolder_TimeI"];\
                PLSca_APredictB_Bootstrap_Sub(Brain_Matrix_train_Mat_Path[0], Behavior_Matrix_train_Mat_Path[0], Covariates_train, Brain_Matrix_test_Mat_Path[0], Behavior_Matrix_test_Mat_Path[0], Covariates_test, Components_Number[0][0], ResultantFolder_TimeI[0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_TimeI + '/Time_' + str(i) + '.log" 2>&1\n'
            script = open(ResultantFolder_TimeI + '/script.sh', 'w');
            script.write(system_cmd);
            script.close();
            Option = ' -V -o "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.o" -e "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.e" ';
            os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh');
            os.system('qsub ' + ResultantFolder_TimeI + '/script.sh ' + Option)

def PLSca_APredictB_Bootstrap_Sub(Brain_Matrix_train_Mat_Path, Behavior_Matrix_train_Mat_Path, Covariates_train, Brain_Matrix_test_Mat_Path, Behavior_Matrix_test_Mat_Path, Covariates_test, Components_Number, ResultantFolder):

    data = sio.loadmat(Brain_Matrix_train_Mat_Path);
    Brain_Matrix_train = data['Brain_Matrix_train'];
    data = sio.loadmat(Behavior_Matrix_train_Mat_Path);
    Behavior_Matrix_train = data['Behavior_Matrix_train'];
    data = sio.loadmat(Brain_Matrix_test_Mat_Path);
    Brain_Matrix_test = data['Brain_Matrix_test'];
    data = sio.loadmat(Behavior_Matrix_test_Mat_Path);
    Behavior_Matrix_test = data['Behavior_Matrix_test'];
    # Using bootstrap to select 66.7% subjects
    SubjectsQuantity = np.shape(Behavior_Matrix_train)[0];
    IndexRange = np.arange(SubjectsQuantity);
    random.shuffle(IndexRange);
    SelectedNumber = round(SubjectsQuantity*0.667);
    IndexRange = IndexRange[:SelectedNumber];
    RemainQuantity = SubjectsQuantity - SelectedNumber;
    Index_Remain = np.random.choice(IndexRange, size = RemainQuantity);
    IndexRange = np.concatenate((IndexRange, Index_Remain));
    Brain_Matrix_train = Brain_Matrix_train[IndexRange,:];
    Behavior_Matrix_train = Behavior_Matrix_train[IndexRange,:];
    Covariates_train = Covariates_train[IndexRange,:];
    PLSca_APredictB(Brain_Matrix_train, Behavior_Matrix_train, Covariates_train, Brain_Matrix_test, Behavior_Matrix_test, Covariates_test, Components_Number, ResultantFolder, 0);

def PLSca_APredictB_Permutation(Brain_Matrix_train, Behavior_Matrix_train, Covariates_train, Brain_Matrix_test, Behavior_Matrix_test, Covariates_test, Components_Number, PermutationTimes, ResultantFolder):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder);
    Brain_Matrix_train_Mat = {'Brain_Matrix_train': Brain_Matrix_train}
    Brain_Matrix_train_Mat_Path = ResultantFolder + '/Brain_Matrix_train.mat'
    sio.savemat(Brain_Matrix_train_Mat_Path, Brain_Matrix_train_Mat);
    Behavior_Matrix_train_Mat = {'Behavior_Matrix_train': Behavior_Matrix_train}
    Behavior_Matrix_train_Mat_Path = ResultantFolder + '/Behavior_Matrix_train.mat'
    sio.savemat(Behavior_Matrix_train_Mat_Path, Behavior_Matrix_train_Mat);

    Brain_Matrix_test_Mat = {'Brain_Matrix_test': Brain_Matrix_test}
    Brain_Matrix_test_Mat_Path = ResultantFolder + '/Brain_Matrix_test.mat'
    sio.savemat(Brain_Matrix_test_Mat_Path, Brain_Matrix_test_Mat);
    Behavior_Matrix_test_Mat = {'Behavior_Matrix_test': Behavior_Matrix_test}
    Behavior_Matrix_test_Mat_Path = ResultantFolder + '/Behavior_Matrix_test.mat'
    sio.savemat(Behavior_Matrix_test_Mat_Path, Behavior_Matrix_test_Mat);

    for i in np.arange(PermutationTimes):
        ResultantFolder_TimeI = ResultantFolder + '/Time_' + str(i)
        if not os.path.exists(ResultantFolder_TimeI):
            os.makedirs(ResultantFolder_TimeI);
        if not os.path.exists(ResultantFolder_TimeI + '/PLS_Prediction.mat'):
            Configuration_Mat = {'Brain_Matrix_train_Mat_Path': Brain_Matrix_train_Mat_Path, \
                                 'Behavior_Matrix_train_Mat_Path': Behavior_Matrix_train_Mat_Path, \
                                 'Covariates_train': Covariates_train, \
                                 'Brain_Matrix_test_Mat_Path': Brain_Matrix_test_Mat_Path, \
                                 'Behavior_Matrix_test_Mat_Path': Behavior_Matrix_test_Mat_Path, \
                                 'Covariates_test': Covariates_test, \
                                 'Components_Number': Components_Number, \
                                 'ResultantFolder_TimeI': ResultantFolder_TimeI};
            sio.savemat(ResultantFolder_TimeI + '/Configuration.mat', Configuration_Mat);

            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + CodesPath + '");\
                from PLSca_CZ_Random_RegressCovariates import PLSca_APredictB_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                Configuration = sio.loadmat("' + ResultantFolder_TimeI + '/Configuration.mat");\
                Brain_Matrix_train_Mat_Path = Configuration["Brain_Matrix_train_Mat_Path"];\
                Behavior_Matrix_train_Mat_Path = Configuration["Behavior_Matrix_train_Mat_Path"];\
                Covariates_train = Configuration["Covariates_train"];\
                Brain_Matrix_test_Mat_Path = Configuration["Brain_Matrix_test_Mat_Path"];\
                Behavior_Matrix_test_Mat_Path = Configuration["Behavior_Matrix_test_Mat_Path"];\
                Covariates_test = Configuration["Covariates_test"];\
                Components_Number = Configuration["Components_Number"];\
                ResultantFolder_TimeI = Configuration["ResultantFolder_TimeI"];\
                PLSca_APredictB_Permutation_Sub(Brain_Matrix_train_Mat_Path[0], Behavior_Matrix_train_Mat_Path[0], Covariates_train, Brain_Matrix_test_Mat_Path[0], Behavior_Matrix_test_Mat_Path[0], Covariates_test, Components_Number[0][0], ResultantFolder_TimeI[0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_TimeI + '/Time_' + str(i) + '.log" 2>&1\n'
            script = open(ResultantFolder_TimeI + '/script.sh', 'w');
            script.write(system_cmd);
            script.close();
            Option = ' -V -o "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.o" -e "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.e" ';
            os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh');
            os.system('qsub ' + ResultantFolder_TimeI + '/script.sh ' + Option)

def PLSca_APredictB_Permutation_Sub(Brain_Matrix_train_Mat_Path, Behavior_Matrix_train_Mat_Path, Covariates_train, Brain_Matrix_test_Mat_Path, Behavior_Matrix_test_Mat_Path, Covariates_test, Components_Number, ResultantFolder):

    data = sio.loadmat(Brain_Matrix_train_Mat_Path);
    Brain_Matrix_train = data['Brain_Matrix_train'];
    data = sio.loadmat(Behavior_Matrix_train_Mat_Path);
    Behavior_Matrix_train = data['Behavior_Matrix_train'];
    data = sio.loadmat(Brain_Matrix_test_Mat_Path);
    Brain_Matrix_test = data['Brain_Matrix_test'];
    data = sio.loadmat(Behavior_Matrix_test_Mat_Path);
    Behavior_Matrix_test = data['Behavior_Matrix_test'];
    PLSca_APredictB(Brain_Matrix_train, Behavior_Matrix_train, Covariates_train, Brain_Matrix_test, Behavior_Matrix_test, Covariates_test, Components_Number, ResultantFolder, 1);

def PLSca_APredictB(Brain_Matrix_train, Behavior_Matrix_train, Covariates_train, Brain_Matrix_test, Behavior_Matrix_test, Covariates_test, Components_Number, ResultantFolder, Permutation_Flag):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder);

    imp_mean = IterativeImputer(random_state=0, sample_posterior=True);
    Behavior_Matrix_train = imp_mean.fit_transform(Behavior_Matrix_train);
    Behavior_Matrix_test = imp_mean.transform(Behavior_Matrix_test);

    Features_Quantity = np.shape(Brain_Matrix_train)[1];
    Behavior_Quantity = np.shape(Behavior_Matrix_train)[1];
    Covariates_Quantity = np.shape(Covariates_train)[1];

    # Controlling covariates from brain data
    df = {};
    for k in np.arange(Covariates_Quantity):
        df['Covariate_'+str(k)] = Covariates_train[:,k];
    # Construct formula
    Formula = 'Data ~ Covariate_0';
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)
    # Regress covariates from each brain feature
    for k in np.arange(Features_Quantity):
        df['Data'] = Brain_Matrix_train[:,k];
        # Regressing covariates using training data
        LinModel_Res = sm.ols(formula=Formula, data=df).fit()
        # Using residuals replace the training data
        Brain_Matrix_train[:,k] = LinModel_Res.resid;
        # Calculating the residuals of testing data by applying the coeffcients of training data
        Coefficients = LinModel_Res.params;
        Brain_Matrix_test[:,k] = Brain_Matrix_test[:,k] - Coefficients[0];
        for m in np.arange(Covariates_Quantity):
            Brain_Matrix_test[:,k] = Brain_Matrix_test[:,k] - Coefficients[m+1]*Covariates_test[:,m]

    # Controlling covariates from behavior data
    df = {};
    for k in np.arange(Covariates_Quantity):
        df['Covariate_'+str(k)] = Covariates_train[:,k];
    # Construct formula
    Formula = 'Data ~ Covariate_0';
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)
    # Regress covariates from each behavior feature
    for k in np.arange(Behavior_Quantity):
        df['Data'] = Behavior_Matrix_train[:,k];
        # Regressing covariates using training data
        LinModel_Res = sm.ols(formula=Formula, data=df).fit()
        # Using residuals replace the training data
        Behavior_Matrix_train[:,k] = LinModel_Res.resid;
        # Calculating the residuals of testing data by applying the coeffcients of training data
        Coefficients = LinModel_Res.params;
        Behavior_Matrix_test[:,k] = Behavior_Matrix_test[:,k] - Coefficients[0];
        for m in np.arange(Covariates_Quantity):
            Behavior_Matrix_test[:,k] = Behavior_Matrix_test[:,k] - Coefficients[m+1]*Covariates_test[:,m]

    if Permutation_Flag:
        # If do permutation, the training scores should be permuted
        Training_Quantity = np.shape(Behavior_Matrix_train)[0]
        Subjects_Index_Random = np.arange(Training_Quantity)
        np.random.shuffle(Subjects_Index_Random)
        Behavior_Matrix_train = Behavior_Matrix_train[Subjects_Index_Random, :]
        PermutationIndex = {'RandomIndex': Subjects_Index_Random}

    normalize = preprocessing.StandardScaler()
    Brain_Matrix_train = normalize.fit_transform(Brain_Matrix_train);
    Brain_Matrix_test = normalize.transform(Brain_Matrix_test);
    Behavior_Matrix_train = normalize.fit_transform(Behavior_Matrix_train)
    Behavior_Matrix_test = normalize.transform(Behavior_Matrix_test)

    plsca = PLSCanonical(n_components=Components_Number, algorithm='svd');
    plsca.fit(Brain_Matrix_train, Behavior_Matrix_train);
    Covariances = plsca.Covariances;
    Brain_test_ca, Behavior_test_ca = plsca.transform(Brain_Matrix_test, Behavior_Matrix_test);

    # PCA
    Training_Quantity = np.shape(Brain_Matrix_train)[0];
    pca = PCA();
    Brain_Matrix_train_PCA = pca.fit_transform(Brain_Matrix_train);
    Brain_PCA_Coeff = pca.components_;
    Brain_PCA_VarianceExplained = pca.explained_variance_ratio_;
    pca = PCA();
    Behavor_Matrix_train_PCA = pca.fit_transform(Behavior_Matrix_train);
    Behavior_PCA_Coeff = pca.components_;
    Behavior_PCA_VarianceExplained = pca.explained_variance_ratio_;

    # Correlation on training data
    Corr_Training = [];
    for k in np.arange(Components_Number):
        Corr_tmp = np.corrcoef(plsca.x_scores_[:,k], plsca.y_scores_[:,k])
        Corr_tmp = Corr_tmp[0,1]
        Corr_Training.append(Corr_tmp)
    # Correlation on testing data
    Corr_Testing = [];
    for k in np.arange(Components_Number):
        Corr_tmp = np.corrcoef(Brain_test_ca[:,k], Behavior_test_ca[:,k])
        Corr_tmp = Corr_tmp[0,1]
        Corr_Testing.append(Corr_tmp)

    Brain_Weight = plsca.x_weights_;
    Behavior_Weight = plsca.y_weights_;
    PLS_result = {'Corr_Training': Corr_Training, \
                  'Corr_Testing': Corr_Testing, 'Covariances': Covariances, \
                  'Brain_test_ca': Brain_test_ca, 'Behavior_test_ca': Behavior_test_ca, \
                  'Brain_Weight': Brain_Weight, 'Behavior_Weight': Behavior_Weight, \
                  'Brain_PCA_Coeff': Brain_PCA_Coeff, 'Behavior_PCA_Coeff': Behavior_PCA_Coeff,
                  'Brain_PCA_VarianceExplained': Brain_PCA_VarianceExplained,
                  'Behavior_PCA_VarianceExplained': Behavior_PCA_VarianceExplained}
    PLS_FileName = 'PLS_Prediction.mat'
    ResultantFile = os.path.join(ResultantFolder, PLS_FileName)
    sio.savemat(ResultantFile, PLS_result)

def PLSca_Pattern_Bootstrap(Brain_Matrix, Behavior_Matrix, Covariates, Components_Number, BootStrapTimes, ResultantFolder):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder);
    Brain_Matrix_Mat = {'Brain_Matrix': Brain_Matrix}
    Brain_Matrix_Mat_Path = ResultantFolder + '/Brain_Matrix.mat'
    sio.savemat(Brain_Matrix_Mat_Path, Brain_Matrix_Mat);
    Behavior_Matrix_Mat = {'Behavior_Matrix': Behavior_Matrix}
    Behavior_Matrix_Mat_Path = ResultantFolder + '/Behavior_Matrix.mat'
    sio.savemat(Behavior_Matrix_Mat_Path, Behavior_Matrix_Mat);

    for i in np.arange(BootStrapTimes):
        ResultantFolder_TimeI = ResultantFolder + '/Time_' + str(i)
        if not os.path.exists(ResultantFolder_TimeI):
            os.makedirs(ResultantFolder_TimeI);
        if not os.path.exists(ResultantFolder_TimeI + '/Weight.mat'):
            Configuration_Mat = {'Brain_Matrix_Mat_Path': Brain_Matrix_Mat_Path, \
                                 'Behavior_Matrix_Mat_Path': Behavior_Matrix_Mat_Path, \
                                 'Covariates': Covariates, \
                                 'Components_Number': Components_Number, \
                                 'ResultantFolder_TimeI': ResultantFolder_TimeI}
            sio.savemat(ResultantFolder_TimeI + '/Configuration.mat', Configuration_Mat);

            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + CodesPath + '");\
                from PLSca_CZ_Random_RegressCovariates import PLSca_Pattern_Bootstrap_Sub;\
                import os;\
                import scipy.io as sio;\
                Configuration = sio.loadmat("' + ResultantFolder_TimeI + '/Configuration.mat");\
                Brain_Matrix_Mat_Path = Configuration["Brain_Matrix_Mat_Path"];\
                Behavior_Matrix_Mat_Path = Configuration["Behavior_Matrix_Mat_Path"];\
                Covariates = Configuration["Covariates"];\
                Components_Number = Configuration["Components_Number"];\
                ResultantFolder_TimeI = Configuration["ResultantFolder_TimeI"];\
                PLSca_Pattern_Bootstrap_Sub(Brain_Matrix_Mat_Path[0], Behavior_Matrix_Mat_Path[0], Covariates, Components_Number[0][0], ResultantFolder_TimeI[0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_TimeI + '/Time_' + str(i) + '.log" 2>&1\n'
            script = open(ResultantFolder_TimeI + '/script.sh', 'w');
            script.write(system_cmd);
            script.close();
            Option = ' -V -o "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.o" -e "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.e" ';
            os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh');
            os.system('qsub ' + ResultantFolder_TimeI + '/script.sh ' + Option)

def PLSca_Pattern_Bootstrap_Sub(Brain_Matrix_Mat_Path, Behavior_Matrix_Mat_Path, Covariates, Components_Number, ResultantFolder):

    data = sio.loadmat(Brain_Matrix_Mat_Path);
    Brain_Matrix = data['Brain_Matrix'];
    data = sio.loadmat(Behavior_Matrix_Mat_Path);
    Behavior_Matrix = data['Behavior_Matrix'];
    # Using bootstrap to select 66.7% subjects
    SubjectsQuantity = np.shape(Behavior_Matrix)[0];
    IndexRange = np.arange(SubjectsQuantity);
    random.shuffle(IndexRange);
    SelectedNumber = round(SubjectsQuantity*0.667);
    IndexRange = IndexRange[:SelectedNumber];
    RemainQuantity = SubjectsQuantity - SelectedNumber;
    Index_Remain = np.random.choice(IndexRange, size = RemainQuantity);
    IndexRange = np.concatenate((IndexRange, Index_Remain));
    Brain_Matrix = Brain_Matrix[IndexRange,:];
    Behavior_Matrix = Behavior_Matrix[IndexRange,:];
    Covariates = Covariates[IndexRange,:];
    PLSca_Pattern(Brain_Matrix, Behavior_Matrix, Covariates, Components_Number, ResultantFolder);

def PLSca_Pattern(Brain_Matrix, Behavior_Matrix, Covariates, Components_Number, ResultantFolder):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)
    imp_mean = IterativeImputer(random_state=0, sample_posterior=True)
    Behavior_Matrix = imp_mean.fit_transform(Behavior_Matrix);

    Features_Quantity = np.shape(Brain_Matrix)[1];
    Covariates_Quantity = np.shape(Covariates)[1];
    Behavior_Quantity = np.shape(Behavior_Matrix)[1];
    # Controlling covariates from brain data
    df = {};
    for k in np.arange(Covariates_Quantity):
        df['Covariate_'+str(k)] = Covariates[:,k];
    # Construct formula
    Formula = 'Data ~ Covariate_0';
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)
    # Regress covariates from each brain feature
    for k in np.arange(Features_Quantity):
        df['Data'] = Brain_Matrix[:,k];
        # Regressing covariates using training data
        LinModel_Res = sm.ols(formula=Formula, data=df).fit()
        # Using residuals replace the training data
        Brain_Matrix[:,k] = LinModel_Res.resid;

    # Controlling covariates from behavior data
    df = {};
    for k in np.arange(Covariates_Quantity):
        df['Covariate_'+str(k)] = Covariates[:,k];
    # Construct formula
    Formula = 'Data ~ Covariate_0';
    for k in np.arange(Covariates_Quantity - 1) + 1:
        Formula = Formula + ' + Covariate_' + str(k)
    # Regress covariates from each brain feature
    for k in np.arange(Behavior_Quantity):
        df['Data'] = Behavior_Matrix[:,k];
        # Regressing covariates using training data
        LinModel_Res = sm.ols(formula=Formula, data=df).fit()
        # Using residuals replace the training data
        Behavior_Matrix[:,k] = LinModel_Res.resid;

    normalize = preprocessing.StandardScaler()
    Brain_Matrix = normalize.fit_transform(Brain_Matrix);
    Behavior_Matrix = normalize.fit_transform(Behavior_Matrix)

    plsca = PLSCanonical(n_components=Components_Number, algorithm='svd');
    plsca.fit(Brain_Matrix, Behavior_Matrix);

    Brain_Scores = plsca.x_scores_;
    Behavior_Scores = plsca.y_scores_;
    Brain_Weights = plsca.x_weights_;
    Behavior_Weights = plsca.y_weights_
    Covariances = plsca.Covariances;

    Weight_Mat = {'Brain_Weights':Brain_Weights, 'Behavior_Weights':Behavior_Weights, 'Brain_Scores':Brain_Scores, 'Behavior_Scores':Behavior_Scores, 'Covariances':Covariances};
    ResultantFile = os.path.join(ResultantFolder, 'Weight.mat')
    sio.savemat(ResultantFile, Weight_Mat)



# -*- coding: utf-8 -*-

import os
import scipy.io as sio
import numpy as np
import time
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import cross_decomposition 
from joblib import Parallel, delayed
import statsmodels.formula.api as sm
CodesPath = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho/scripts/AnalysisScripts/Functions';
 
def PLSr2_KFold_RandomCV_MultiTimes(Brain_Matrix, Behavior_Matrix, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity, Permutation_Flag, Queue):
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder);

    Brain_Matrix_Mat = {'Brain_Matrix': Brain_Matrix}
    Brain_Matrix_Mat_Path = ResultantFolder + '/Brain_Matrix.mat'
    sio.savemat(Brain_Matrix_Mat_Path, Brain_Matrix_Mat);

    Finish_File = [];
    Corr_MTimes = np.zeros(CVRepeatTimes);
    MAE_MTimes = np.zeros(CVRepeatTimes);
    for i in np.arange(CVRepeatTimes):
        ResultantFolder_TimeI = ResultantFolder + '/Time_' + str(i)
        if not os.path.exists(ResultantFolder_TimeI):
            os.makedirs(ResultantFolder_TimeI);

        if not os.path.exists(ResultantFolder_TimeI + '/Res_NFold.mat'):
            Configuration_Mat = {'Brain_Matrix_Mat_Path': Brain_Matrix_Mat_Path, \
                                 'Behavior_Matrix': Behavior_Matrix, \
                                 'Covariates': Covariates, \
                                 'Fold_Quantity': Fold_Quantity, \
                                 'ComponentNumber_Range': ComponentNumber_Range, \
                                 'CVRepeatTimes': CVRepeatTimes, \
                                 'ResultantFolder_TimeI': ResultantFolder_TimeI, \
                                 'Parallel_Quantity': Parallel_Quantity, \
                                 'Permutation_Flag': Permutation_Flag};
            sio.savemat(ResultantFolder_TimeI + '/Configuration.mat', Configuration_Mat);

            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + CodesPath + '");\
                from PLSr2_CZ_Random_RegressCovariates import PLSr2_KFold_RandomCV_MultiTimes_Sub; \
                import os;\
                import scipy.io as sio;\
                Configuration = sio.loadmat("' + ResultantFolder_TimeI + '/Configuration.mat");\
                Brain_Matrix_Mat_Path = Configuration["Brain_Matrix_Mat_Path"];\
                Behavior_Matrix = Configuration["Behavior_Matrix"];\
                Covariates = Configuration["Covariates"];\
                Fold_Quantity = Configuration["Fold_Quantity"];\
                ComponentNumber_Range = Configuration["ComponentNumber_Range"];\
                ResultantFolder_TimeI = Configuration["ResultantFolder_TimeI"];\
                Permutation_Flag = Configuration["Permutation_Flag"];\
                Parallel_Quantity = Configuration["Parallel_Quantity"];\
                PLSr2_KFold_RandomCV_MultiTimes_Sub(Brain_Matrix_Mat_Path[0], Behavior_Matrix, Covariates, Fold_Quantity[0][0], ComponentNumber_Range[0], 20, ResultantFolder_TimeI[0], Parallel_Quantity[0][0], Permutation_Flag[0][0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_TimeI + '/Time_' + str(i) + '.log" 2>&1\n'
            Finish_File.append(ResultantFolder_TimeI + 'Res_NFold.mat');
            script = open(ResultantFolder_TimeI + '/script.sh', 'w');
            script.write(system_cmd);
            script.close();
        # Submit jobs
        #Option = ' -q ' + Queue + ' -V -o "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.o" -e "' + ResultantFolder_TimeI + '/RandomCV_' + str(i) + '.e" ';
        #os.system('qsub -l h_vmem=5G,s_vmem=5G -N RandomCV_' + str(i) + Option + ResultantFolder_TimeI + '/script.sh')  
            os.system('chmod +x ' + ResultantFolder_TimeI + '/script.sh');
            Option = ' -V -o "' + ResultantFolder_TimeI + '/perm_' + str(i) + '.o" -e "' + ResultantFolder_TimeI + '/perm_' + str(i) + '.e" ';
            os.system('qsub -q ' + Queue + ' -N perm_' + str(i) + Option + ResultantFolder_TimeI + '/script.sh')

def PLSr2_KFold_RandomCV_MultiTimes_Sub(Brain_Matrix_Mat_Path, Behavior_Matrix, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity, Permutation_Flag):
    data = sio.loadmat(Brain_Matrix_Mat_Path);
    Brain_Matrix = data['Brain_Matrix'];
    PLSr2_KFold_RandomCV(Brain_Matrix, Behavior_Matrix, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity, Permutation_Flag);

def PLSr2_KFold_RandomCV(Brain_Matrix, Behavior_Matrix, Covariates, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity, Permutation_Flag):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)

    Subjects_Quantity = np.shape(Behavior_Matrix)[0]
    EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    RandIndex = np.arange(Subjects_Quantity)
    np.random.shuffle(RandIndex)
    
    Fold_Corr = [];
    Fold_MAE = [];
    Fold_Weight = [];

    Features_Quantity = np.shape(Brain_Matrix)[1];
    Behavior_Quantity = np.shape(Behavior_Matrix)[1];
    Covariates_Quantity = np.shape(Covariates)[1];
    Fold_Corr = np.zeros((Fold_Quantity, Behavior_Quantity))
    for j in np.arange(Fold_Quantity):
        Fold_J_Index = RandIndex[EachFold_Size * j + np.arange(EachFold_Size)]
        if Remain > j:
            Fold_J_Index = np.insert(Fold_J_Index, len(Fold_J_Index), RandIndex[EachFold_Size * Fold_Quantity + j]);

        Brain_Matrix_test = Brain_Matrix[Fold_J_Index, :]
        Behavior_Matrix_test = Behavior_Matrix[Fold_J_Index, :]
        Covariates_test = Covariates[Fold_J_Index, :]
        Brain_Matrix_train = np.delete(Brain_Matrix, Fold_J_Index, axis=0)
        Behavior_Matrix_train = np.delete(Behavior_Matrix, Fold_J_Index, axis=0) 
        Covariates_train = np.delete(Covariates, Fold_J_Index, axis=0)

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

        if Permutation_Flag:
            # If do permutation, the training scores should be permuted, while the testing scores remain
            Subjects_Index_Random = np.arange(np.shape(Behavior_Matrix_train)[0])
            np.random.shuffle(Subjects_Index_Random)
            Behavior_Matrix_train = Behavior_Matrix_train[Subjects_Index_Random, :]
            if j == 0:
                PermutationIndex = {'Fold_0': Subjects_Index_Random}
            else:
                PermutationIndex['Fold_' + str(j)] = Subjects_Index_Random
        
        Optimal_ComponentNumber, Inner_Corr, Inner_MAE_inv = PLSr2_OptimalComponentNumber_KFold(Brain_Matrix_train, Behavior_Matrix_train, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes_ForInner, ResultantFolder, Parallel_Quantity)

        clf = cross_decomposition.PLSRegression(n_components = Optimal_ComponentNumber)
        clf.fit(Brain_Matrix_train, Behavior_Matrix_train)
        Fold_J_Score = clf.predict(Brain_Matrix_test)

        # Correlation on testing data
        Fold_J_Corr = [];
        for k in np.arange(Behavior_Quantity):
            Fold_J_Corr_tmp = np.corrcoef(Fold_J_Score[:,k], Behavior_Matrix_test[:,k]);
            Fold_J_Corr_tmp = Fold_J_Corr_tmp[0,1]
            Fold_J_Corr.append(Fold_J_Corr_tmp)
       
        Coef = clf.coef_;
        Fold_J_result = {'Index':Fold_J_Index, 'Test_Score':Brain_Matrix_test, 'Predict_Score':Fold_J_Score, 'Corr':Fold_J_Corr, 'ComponentNumber':Optimal_ComponentNumber, 'Weight': Coef}
        Fold_J_FileName = 'Fold_' + str(j) + '_Score.mat'
        ResultantFile = os.path.join(ResultantFolder, Fold_J_FileName)
        sio.savemat(ResultantFile, Fold_J_result)
        Fold_Corr[j,:] = Fold_J_Corr;

    Mean_Corr = np.mean(Fold_Corr, axis = 0)
    Res_NFold = {'Mean_Corr':Mean_Corr};
    ResultantFile = os.path.join(ResultantFolder, 'Res_NFold.mat')
    sio.savemat(ResultantFile, Res_NFold)
    
    if Permutation_Flag:
        sio.savemat(ResultantFolder + '/PermutationIndex.mat', PermutationIndex)

    return (Mean_Corr)  

def PLSr2_OptimalComponentNumber_KFold(Training_Brain_Matrix, Training_Behavior_Matrix, Fold_Quantity, ComponentNumber_Range, CVRepeatTimes, ResultantFolder, Parallel_Quantity):
   
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder);
 
    Subjects_Quantity = np.shape(Training_Behavior_Matrix)[0]
    Inner_EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)));
    Remain = np.mod(Subjects_Quantity, Fold_Quantity);    

    Inner_Corr_Mean = np.zeros((CVRepeatTimes, len(ComponentNumber_Range)))
    Inner_MAE_inv_Mean = np.zeros((CVRepeatTimes, len(ComponentNumber_Range)))
    ComponentNumber_Quantity = len(ComponentNumber_Range)
    for i in np.arange(CVRepeatTimes):
        
        RandIndex = np.arange(Subjects_Quantity)
        np.random.shuffle(RandIndex)

        Inner_Corr = np.zeros((Fold_Quantity, len(ComponentNumber_Range)))
        Inner_MAE_inv = np.zeros((Fold_Quantity, len(ComponentNumber_Range)))
        ComponentNumber_Quantity = len(ComponentNumber_Range)

        for k in np.arange(Fold_Quantity):

            Inner_Fold_K_Index = RandIndex[Inner_EachFold_Size * k + np.arange(Inner_EachFold_Size)]
            if Remain > k:
                Inner_Fold_K_Index = np.insert(Inner_Fold_K_Index, len(Inner_Fold_K_Index), RandIndex[Inner_EachFold_Size * Fold_Quantity + k])

            Inner_Fold_K_Data_test = Training_Brain_Matrix[Inner_Fold_K_Index, :]
            Inner_Fold_K_Score_test = Training_Behavior_Matrix[Inner_Fold_K_Index, :]
            Inner_Fold_K_Data_train = np.delete(Training_Brain_Matrix, Inner_Fold_K_Index, axis = 0)
            Inner_Fold_K_Score_train = np.delete(Training_Behavior_Matrix, Inner_Fold_K_Index, axis = 0);

            Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(PLSr2_SubComponentNumber)(Inner_Fold_K_Data_train, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test, Inner_Fold_K_Score_test, ComponentNumber_Range[l], l, ResultantFolder) for l in np.arange(len(ComponentNumber_Range)))        
        
            for l in np.arange(ComponentNumber_Quantity):
                print(l)
                ComponentNumber_l_Mat_Path = ResultantFolder + '/ComponentNumber_' + str(l) + '.mat';
                ComponentNumber_l_Mat = sio.loadmat(ComponentNumber_l_Mat_Path)
                Inner_Corr[k, l] = ComponentNumber_l_Mat['Corr'][0][0]
                Inner_MAE_inv[k, l] = ComponentNumber_l_Mat['MAE_inv']
                os.remove(ComponentNumber_l_Mat_Path)
            
            Inner_Corr = np.nan_to_num(Inner_Corr)

        Inner_Corr_Mean[i, :] = np.mean(Inner_Corr, axis = 0)
        Inner_MAE_inv_Mean[i, :] = np.mean(Inner_MAE_inv, axis = 0)

    Inner_Corr_CVMean = np.mean(Inner_Corr_Mean, axis = 0)
    Inner_MAE_inv_CVMean = np.mean(Inner_MAE_inv_Mean, axis = 0)
    Inner_Corr_CVMean = (Inner_Corr_CVMean - np.mean(Inner_Corr_CVMean)) / np.std(Inner_Corr_CVMean)
    Inner_MAE_inv_CVMean = (Inner_MAE_inv_CVMean - np.mean(Inner_MAE_inv_CVMean)) / np.std(Inner_MAE_inv_CVMean)
    Inner_Evaluation = Inner_Corr_CVMean + Inner_MAE_inv_CVMean
    
    Inner_Evaluation_Mat = {'Inner_Corr_Mean':Inner_Corr_Mean, 'Inner_MAE_inv_Mean':Inner_MAE_inv_Mean, 'Inner_Corr_CVMean': Inner_Corr_CVMean, 'Inner_MAE_inv_CVMean': Inner_MAE_inv_CVMean, 'Inner_Evaluation':Inner_Evaluation}
    sio.savemat(ResultantFolder + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
    Optimal_ComponentNumber_Index = np.argmax(Inner_Evaluation) 
    Optimal_ComponentNumber = ComponentNumber_Range[Optimal_ComponentNumber_Index]
    return (Optimal_ComponentNumber, Inner_Corr, Inner_MAE_inv)

def PLSr2_SubComponentNumber(Training_Brain_Matrix, Training_Behavior_Matrix, Testing_Brain_Matrix, Testing_Behavior_Matrix, ComponentNumber, ComponentNumber_ID, ResultantFolder):
    clf = cross_decomposition.PLSRegression(n_components=ComponentNumber)
    clf.fit(Training_Brain_Matrix, Training_Behavior_Matrix)
    Predict_Score = clf.predict(Testing_Brain_Matrix)
    Corr_All = [];
    MAE_inv_All = [];
    Behavior_Quantity = np.shape(Testing_Behavior_Matrix)[1];
    for k in np.arange(Behavior_Quantity):
        Corr = np.corrcoef(Predict_Score[:,k], Testing_Behavior_Matrix[:,k]);
        Corr = Corr[0,1]
        Corr_All.append(Corr);
        MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score[:,k] - Testing_Behavior_Matrix[:,k])) / np.abs(np.sum(Testing_Behavior_Matrix[:,k])));
        MAE_inv_All.append(MAE_inv);

    Corr_All_Sum = np.sum(Corr_All);
    MAE_inv_All_Sum = np.sum(MAE_inv_All);
    result = {'Corr': Corr_All_Sum, 'MAE_inv':MAE_inv_All_Sum}
    ResultantFile = ResultantFolder + '/ComponentNumber_' + str(ComponentNumber_ID) + '.mat'
    sio.savemat(ResultantFile, result)
    

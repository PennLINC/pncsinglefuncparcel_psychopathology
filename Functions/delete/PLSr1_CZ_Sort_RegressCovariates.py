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

def PLSr1_KFold_Sort_Permutation(Subjects_Data, Subjects_Score, Covariates, Times_IDRange, Fold_Quantity, ComponentNumber_Range, ResultantFolder, Parallel_Quantity, Max_Queued, Queue, Permutation_RandIndex_File_List=''):
    
    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)
    Subjects_Data_Mat = {'Subjects_Data': Subjects_Data}
    Subjects_Data_Mat_Path = ResultantFolder + '/Subjects_Data.mat'
    sio.savemat(Subjects_Data_Mat_Path, Subjects_Data_Mat)
    Finish_File = []
    Times_IDRange_Todo = np.int64(np.array([]))
    for i in np.arange(len(Times_IDRange)):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange[i])
        if not os.path.exists(ResultantFolder_I):
            os.makedirs(ResultantFolder_I)
        if not os.path.exists(ResultantFolder_I + '/Res_NFold.mat'):
            Times_IDRange_Todo = np.insert(Times_IDRange_Todo, len(Times_IDRange_Todo), Times_IDRange[i])

            if Permutation_RandIndex_File_List != '':
              Permutation_RandIndex_File = Permutation_RandIndex_File_List[i]
            else:
              Permutation_RandIndex_File = '';
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score': Subjects_Score, 'Covariates': Covariates, 'Fold_Quantity': Fold_Quantity, \
                'ComponentNumber_Range': ComponentNumber_Range, 'ResultantFolder_I': ResultantFolder_I, 'Parallel_Quantity': Parallel_Quantity, 'Permutation_RandIndex_File': Permutation_RandIndex_File};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("' + CodesPath + '");\
                from PLSr1_CZ_Sort_RegressCovariates import PLSr1_KFold_Sort_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score = configuration["Subjects_Score"];\
                Covariates = configuration["Covariates"];\
                Fold_Quantity = configuration["Fold_Quantity"];\
                ComponentNumber_Range = configuration["ComponentNumber_Range"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                Permutation_RandIndex_File = configuration["Permutation_RandIndex_File"];\
                Parallel_Quantity = configuration["Parallel_Quantity"];\
                PLSr1_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score[0], Covariates, Fold_Quantity[0][0], ComponentNumber_Range[0], ResultantFolder_I[0], Parallel_Quantity[0][0], Permutation_RandIndex_File)\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/perm_' + str(Times_IDRange[i]) + '.log" 2>&1\n'
            Finish_File.append(ResultantFolder_I + '/Res_NFold.mat')
            script = open(ResultantFolder_I + '/script.sh', 'w')  
            script.write(system_cmd)
            script.close()

    if len(Times_IDRange_Todo) > Max_Queued:
        Submit_First_Quantity = Max_Queued
    else:
        Submit_First_Quantity = len(Times_IDRange_Todo)
    for i in np.arange(Submit_First_Quantity):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[i])
        Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.e" ';
        os.system('qsub -l h_vmem=5G,s_vmem=5G -q ' + Queue + ' -N perm_' + str(Times_IDRange_Todo[i]) + Option + ResultantFolder_I + '/script.sh')
    if len(Times_IDRange_Todo) > Max_Queued:
        Finished_Quantity = 0;
        while 1:
            for i in np.arange(len(Finish_File)):
                if os.path.exists(Finish_File[i]):
                    Finished_Quantity = Finished_Quantity + 1
                    print(Finish_File[i])            
                    del(Finish_File[i])
                    print(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                    print('Finish quantity = ' + str(Finished_Quantity))
                    time.sleep(8)
                    ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[Max_Queued + Finished_Quantity - 1])
                    Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Max_Queued + Finished_Quantity - 1]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Max_Queued + Finished_Quantity - 1]) + '.e" ';
                    cmd = 'qsub -l h_vmem=5G,s_vmem=5G -q ' + Queue + ' -N perm_' + str(Times_IDRange_Todo[Max_Queued + Finished_Quantity - 1]) + Option + ResultantFolder_I + '/script.sh'
                    # print(cmd)
                    os.system(cmd)
                    break
            if len(Finish_File) == 0:
                break
            if Max_Queued + Finished_Quantity >= len(Finish_File):
                break

def PLSr1_KFold_Sort_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, ResultantFolder, Parallel_Quantity, RandIndex_File=''):
    data = sio.loadmat(Subjects_Data_Mat_Path);
    Subjects_Data = data['Subjects_Data'];
    PLSr1_KFold_Sort(Subjects_Data, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, ResultantFolder, Parallel_Quantity, 1, RandIndex_File);

def PLSr1_KFold_Sort(Subjects_Data, Subjects_Score, Covariates, Fold_Quantity, ComponentNumber_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag, RandIndex_File=''):

    if not os.path.exists(ResultantFolder):
        os.makedirs(ResultantFolder)
    Subjects_Quantity = len(Subjects_Score)
    # Sort the subjects score
    Sorted_Index = np.argsort(Subjects_Score)
    Subjects_Data = Subjects_Data[Sorted_Index, :]
    Subjects_Score = Subjects_Score[Sorted_Index]
    Covariates = Covariates[Sorted_Index]

    EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    MaxSize = EachFold_Size * Fold_Quantity
    EachFold_Max = np.ones(Fold_Quantity, np.int) * MaxSize
    tmp = np.arange(Fold_Quantity - 1, -1, -1)
    EachFold_Max = EachFold_Max - tmp;
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    for j in np.arange(Remain):
        EachFold_Max[j] = EachFold_Max[j] + Fold_Quantity

    Fold_Corr = [];
    Fold_MAE = [];
    Fold_Weight = [];
    
    Features_Quantity = np.shape(Subjects_Data)[1];
    for j in np.arange(Fold_Quantity):

        Fold_J_Index = np.arange(j, EachFold_Max[j], Fold_Quantity)
        Subjects_Data_test = Subjects_Data[Fold_J_Index, :]
        Subjects_Score_test = Subjects_Score[Fold_J_Index]
        Covariates_test = Covariates[Fold_J_Index]
        Subjects_Data_train = np.delete(Subjects_Data, Fold_J_Index, axis=0)
        Subjects_Score_train = np.delete(Subjects_Score, Fold_J_Index)
        Covariates_train = np.delete(Covariates, Fold_J_Index, axis=0)

        Covariates_Quantity = np.shape(Covariates)[1];
        # Controlling covariates from brain data
        df = {};
        for k in np.arange(Covariates_Quantity):
            df['Covariate_'+str(k)] = Covariates_train[:,k];
        # Construct formula
        Formula = 'Data ~ Covariate_0';
        for k in np.arange(Covariates_Quantity - 1) + 1:
            Formula = Formula + ' + Covariate_' + str(k)
        # Regress covariates from each brain features 
        for k in np.arange(Features_Quantity):
            df['Data'] = Subjects_Data_train[:,k];
            # Regressing covariates using training data
            LinModel_Res = sm.ols(formula=Formula, data=df).fit()
            # Using residuals replace the training data
            Subjects_Data_train[:,k] = LinModel_Res.resid;
            # Calculating the residuals of testing data by applying the coeffcients of training data
            Coefficients = LinModel_Res.params;
            Subjects_Data_test[:,k] = Subjects_Data_test[:,k] - Coefficients[0];
            for m in np.arange(Covariates_Quantity):
                Subjects_Data_test[:, k] = Subjects_Data_test[:,k] - Coefficients[m+1]*Covariates_test[:,m]

        if Permutation_Flag:
            # If do permutation, the training scores should be permuted, while the testing scores remain
            Subjects_Index_Random = np.arange(len(Subjects_Score_train))
            np.random.shuffle(Subjects_Index_Random)
            Subjects_Score_train = Subjects_Score_train[Subjects_Index_Random]
            if j == 0:
                PermutationIndex = {'Fold_0': Subjects_Index_Random}
            else:
                PermutationIndex['Fold_' + str(j)] = Subjects_Index_Random
        
        Optimal_ComponentNumber, Inner_Corr, Inner_MAE_inv = PLSr1_OptimalComponentNumber_KFold(Subjects_Data_train, Subjects_Score_train, Fold_Quantity, ComponentNumber_Range, ResultantFolder, Parallel_Quantity)

        normalize = preprocessing.StandardScaler()
        Subjects_Data_train = normalize.fit_transform(Subjects_Data_train);
        Subjects_Data_test = normalize.transform(Subjects_Data_test);

        clf = cross_decomposition.PLSRegression(n_components = Optimal_ComponentNumber)
        clf.fit(Subjects_Data_train, Subjects_Score_train)
        Fold_J_Score = clf.predict(Subjects_Data_test)
        Fold_J_Score = np.transpose(Fold_J_Score)[0]

        Fold_J_Corr = np.corrcoef(Fold_J_Score, Subjects_Score_test)
        Fold_J_Corr = Fold_J_Corr[0,1]
        Fold_Corr.append(Fold_J_Corr)
        Fold_J_MAE = np.mean(np.abs(np.subtract(Fold_J_Score,Subjects_Score_test)))
        Fold_MAE.append(Fold_J_MAE)

        Weight = clf.coef_ / np.sqrt(np.sum(clf.coef_ **2));

        Fold_J_result = {'Index':Fold_J_Index, 'Test_Score':Subjects_Score_test, 'Predict_Score':Fold_J_Score, 'Corr':Fold_J_Corr, 'MAE':Fold_J_MAE, 'ComponentNumber':Optimal_ComponentNumber, 'Inner_Corr':Inner_Corr, 'Inner_MAE_inv':Inner_MAE_inv, 'w_Brain':Weight}
        Fold_J_FileName = 'Fold_' + str(j) + '_Score.mat'
        ResultantFile = os.path.join(ResultantFolder, Fold_J_FileName)
        sio.savemat(ResultantFile, Fold_J_result)

    Fold_Corr = [0 if np.isnan(x) else x for x in Fold_Corr]
    Mean_Corr = np.mean(Fold_Corr)
    Mean_MAE = np.mean(Fold_MAE)
    Res_NFold = {'Mean_Corr':Mean_Corr, 'Mean_MAE':Mean_MAE};
    ResultantFile = os.path.join(ResultantFolder, 'Res_NFold.mat')
    sio.savemat(ResultantFile, Res_NFold)
    
    if Permutation_Flag:
        sio.savemat(ResultantFolder + '/PermutationIndex.mat', PermutationIndex)

    return (Mean_Corr, Mean_MAE)  

def PLSr1_OptimalComponentNumber_KFold(Training_Data, Training_Score, Fold_Quantity, ComponentNumber_Range, ResultantFolder, Parallel_Quantity):

    Subjects_Quantity = len(Training_Score)
    Sorted_Index = np.argsort(Training_Score)
    Training_Data = Training_Data[Sorted_Index, :]
    Training_Score = Training_Score[Sorted_Index]
    
    Inner_EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    MaxSize = Inner_EachFold_Size * Fold_Quantity
    EachFold_Max = np.ones(Fold_Quantity, np.int) * MaxSize
    tmp = np.arange(Fold_Quantity - 1, -1, -1)
    EachFold_Max = EachFold_Max - tmp
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    for j in np.arange(Remain):
    	EachFold_Max[j] = EachFold_Max[j] + Fold_Quantity
    
    print(ComponentNumber_Range);
    Inner_Corr = np.zeros((Fold_Quantity, len(ComponentNumber_Range)))
    Inner_MAE_inv = np.zeros((Fold_Quantity, len(ComponentNumber_Range)))
    ComponentNumber_Quantity = len(ComponentNumber_Range)
    for k in np.arange(Fold_Quantity):
        
        Inner_Fold_K_Index = np.arange(k, EachFold_Max[k], Fold_Quantity)
        Inner_Fold_K_Data_test = Training_Data[Inner_Fold_K_Index, :]
        Inner_Fold_K_Score_test = Training_Score[Inner_Fold_K_Index]
        Inner_Fold_K_Data_train = np.delete(Training_Data, Inner_Fold_K_Index, axis=0)
        Inner_Fold_K_Score_train = np.delete(Training_Score, Inner_Fold_K_Index)
        Scale = preprocessing.MinMaxScaler()
        Inner_Fold_K_Data_train = Scale.fit_transform(Inner_Fold_K_Data_train)
        Inner_Fold_K_Data_test = Scale.transform(Inner_Fold_K_Data_test)    
        
        Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(PLSr1_SubComponentNumber)(Inner_Fold_K_Data_train, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test, Inner_Fold_K_Score_test, ComponentNumber_Range[l], l, ResultantFolder) for l in np.arange(len(ComponentNumber_Range)))        
        
        for l in np.arange(ComponentNumber_Quantity):
            print(l)
            ComponentNumber_l_Mat_Path = ResultantFolder + '/ComponentNumber_' + str(l) + '.mat';
            ComponentNumber_l_Mat = sio.loadmat(ComponentNumber_l_Mat_Path)
            Inner_Corr[k, l] = ComponentNumber_l_Mat['Corr'][0][0]
            Inner_MAE_inv[k, l] = ComponentNumber_l_Mat['MAE_inv']
            os.remove(ComponentNumber_l_Mat_Path)
            
        Inner_Corr = np.nan_to_num(Inner_Corr)
    Inner_Corr_Mean = np.mean(Inner_Corr, axis=0)
    Inner_Corr_Mean = (Inner_Corr_Mean - np.mean(Inner_Corr_Mean)) / np.std(Inner_Corr_Mean)
    Inner_MAE_inv_Mean = np.mean(Inner_MAE_inv, axis=0)
    Inner_MAE_inv_Mean = (Inner_MAE_inv_Mean - np.mean(Inner_MAE_inv_Mean)) / np.std(Inner_MAE_inv_Mean)
    Inner_Evaluation = Inner_Corr_Mean + Inner_MAE_inv_Mean

    Inner_Evaluation_Mat = {'Inner_Corr':Inner_Corr, 'Inner_MAE_inv':Inner_MAE_inv, 'Inner_Evaluation':Inner_Evaluation}
    sio.savemat(ResultantFolder + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
    Optimal_ComponentNumber_Index = np.argmax(Inner_Evaluation) 
    Optimal_ComponentNumber = ComponentNumber_Range[Optimal_ComponentNumber_Index]
    return (Optimal_ComponentNumber, Inner_Corr, Inner_MAE_inv)

def PLSr1_SubComponentNumber(Training_Data, Training_Score, Testing_Data, Testing_Score, ComponentNumber, ComponentNumber_ID, ResultantFolder):
    clf = cross_decomposition.PLSRegression(n_components=ComponentNumber)
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)
    Predict_Score = np.transpose(Predict_Score)[0]
    Corr = np.corrcoef(Predict_Score, Testing_Score)
    Corr = Corr[0,1]
    MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score - Testing_Score)))
    result = {'Corr': Corr, 'MAE_inv':MAE_inv}
    ResultantFile = ResultantFolder + '/ComponentNumber_' + str(ComponentNumber_ID) + '.mat'
    sio.savemat(ResultantFile, result)
    



# Linking Individual Differences in Personalized Functional Network Topography to Psychopathology in Youth

The spatial distribution of large-scale functional networks across association cortex is refined in neurodevelopment and is related to individual differences in cognition. However, it remains unknown if individual variability in this functional topography is associated with major dimensions of psychopathology in youth. Capitalizing on advances in machine learning and a large sample imaged with 27 min of high-quality functional MRI (fMRI) data (n = 790, ages 8-23 years), we examined the association between the functional topography and dimensions of psychopathology in youth. We found the topographies of individualized functional networks significantly predicted the four correlated dimensions of psychopathology, including fear, psychosis, externalizing and anxious-misery. The contribution patterns of the four dimensions were similar: the .  Further analysis revealed that individualized functional topography predicted the general psychopathology factor (r = 0.15, p(perm)<0.001), which underlies the significant predictions of the four correlated dimensions. There results provide novel evidence that individual differences in person-specific functional topography in association networks are linked to dimensions of psychopathology in youth. 

### Project Lead
Zaixu Cui

### Faculty Leads
Theodore D. Satterthwaite

### Analytic Replicator
Adam R. Pines

### Collaborators 
Adam R. Pines, Hongming Li, Tyler M. Moore, Azeez Adebimpe, Jacob W. Vogel, Sheila Shanmugan, Bart Larsen, Max Bertolero, Cedric H. Xia, Raquel E. Gur, Ruben C. Gur, Desmond J. Oathes, Aaron F. Alexander-Bloch, Michael P. Milham, Giovanni A. Salum, Monica E. Calkins, David R. Roalf, Russell T. Shinohara, Daniel H. Wolf, Christos Davatzikos, Danielle S. Bassett, Damien A. Fair, Yong Fan

### Current Project Status
In preparation.

### Datasets
PNC dataset:<https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000607.v3.p2>

### Github Repository
<https://github.com/PennLINC/pncsinglefuncparcel_psychopathology>

### Path to Data on Filesystem 
/cbica/projects/pncSingleFuncParcel/pncSingleFuncParcel_psycho

<br>

# CODE DOCUMENTATION
The steps below detail how to replicate all aspects of this project, from neuroimage processing to statistical anlysis and figure generation.

### Step_1st_PrepareData
1. Step_1st_SubjectFilter.R: 
   > Select our sample (i.e., 790 subjects). Inclusion criteria: healthExcludev2 = 0; fsFinalExclude = 0; restExclude = 0; restExcludeVoxelwise = 0; nbackFcExclude = 0; nbackFcExcludeVoxelwise = 0; idemoFcExclude = 0; idemoFcExcludeVoxelwise = 0. Finally, a sample of 790 subjects was created.
2. Step_2nd_ExtractBehavior.R
   > Extract correlated dimensions, bifactors, and item-level symptom of psychopathology.
3. Step_3rd_CopyStructFSFiles.R, Step_4th_DataFSProcessing.m, Step_5th_MergeModalities.m: 
   > Project fmri data into surface and then combine the three modalities.
4. Step_6th_tSNRMask*: 
   > Generate the tSNR mask.
5. Step_7th_EFA.R, Step_8th_CFA: 
   > Plot figures for exploratory factor analysis (Figure 2) and confirmatory factor analysis (Figure 5).
6. Step_9th_Prevelance.R: 
   > Calculate the prevelance of disorders.

### Step_2nd_SingleParcellation
Step1 to step4 are codes for single functional parcellation (Li et al., 2017, NeuroImage). See (https://github.com/hmlicas/Collaborative_Brain_Decomposition) for the toolbox of single parcellation.
1. Step_1st_CreatePrepData.m: 
   > Generate the spatial neighborhood for fsaverage5 surface space. After removing medial wall, we have 18715 vertices.
2. Step_2nd_ParcellationInitialize.m:
   > Calculate the group parcellation, which will be the initialization of single subject parcellation. We randomly chose 100 subjects and combined these subjects' data along time points and run non-negative matrix factorization (NMF) to decompose the whole brain into 17 networks. We repeated this procedure 50 times, finally 50 group atlas was acquired.
3. Step_3rd_SelRobustInit.m:
   > Use normalized cuts based spectrum clustering method to cluster the acquired 50 group atlases. Finally, one group atlas with 17 networks was acquired.
4. Step_4th_IndividualParcel.m:
   > Based on the acquired group atlas and the subject's specific fMRI data, we calculated the atlas for this specific subject. Finally, each subject had a loading matrix, in which the loading value quantifies the probability each vertex belonging to each network.
5. Step_5th_AtlasInformation_Extract.m:
   > Extract all subjects' loading matrix information into the same folder. Also, extracting the discrete atlas for each subject by categorizing each vertex to the network with the highest loading, and put them into the same folder.
6. Step_6th_GroupAtlas_Extract.m:
   > Extract loading matrix group atlas (the output of the script 'Step_3rd_SelRobustInit.m') and create the group discrete atlas by categorizing each vertex to the network with the highest loading.
7. Step_7th_NetworkNaming_Yeo.m:
   > Calculate the overlap between our networks with Yeo atlas.
8. Step_8th_Visualize_Workbench_Atlas.m:
   > Calculate inter-subjects variability of loadings according to the formula: MADM=median(|x(i) - median(x)|). Calculating the average of 17 variability brain map.
9. Step_9th_Visualize_Workbench_AtlasVariability.m:
   > Visualize the atlas variability map using workbench.
10. Step_10th_ViolinPlot_AtlasVariability_Loading.R:
   > Visualize the variability of each network using violin plot.

### Step_3rd_Psychopathology_SaveMat.R
   > Saving age, sex, motion, four correlated dimensions of psychopathology, five bifactors of psychopathology, and 112 item-level psychopathology symptoms to .mat file.
   
### Step_4th_AtlasFeature_SaveMat.m
   > For each participant, extracting all the loadings of the 17 networks to be a high-dimensional feature vector. 
   
### Step_5th_PLSr1_Corrtraits
Prediction of the four correlated dimensions of psychopathology using personalized functional topography.
1. Step_1st_Prediction_Fear_RandomCV.py, Step_1st_Prediction_Psychosis_RandomCV.py, Step_1st_Prediction_Externalizing_RandomCV.py, Step_1st_Prediction_Mood_RandomCV.py:
   > Predict factor scores of fear, psychosis, externalizing and mood/anxious-misery dimensions using networks loading features acquired in Step 4th. Repeated (i.e., 101 times) two-fold cross validation (2F-CV) was used here. Partial least square regression was used to predict the behavior. We applied permutation testing (i.e., 1,000 times) to evaluate the significance of the prediction accuracy.
2. Step_2nd_Sig.m:
   > Extract the prediction results and calculate the P value of the prediction accuracy based on the distribution from the permutation testing.
3. Step_3rd_Scatter_Fear.R, Step_3rd_Scatter_Psychosis.R, Step_3rd_Scatter_Externalizing.R, Step_3rd_Scatter_Mood.R:
   > Plot the scatter plot of the correlation between actual and predicted scores of the prediction of the four correlated dimensions of psychopathology. The 2F-CV was repeated 101 times and we reported the one with the median prediction accuracy.
4. Step_4th_Prediction_RandomCV_All_Violin.R:
   > Plot the actual prediction accuracy (i.e., correlation r between actual and predicted scores) and boxplot/violin plot of the permutation distribution.
5. Step_5th_Weight_Visualize_Workbench_Fear_RandomCV.m, Step_5th_Weight_Visualize_Workbench_Psychosis_RandomCV.m, Step_5th_Weight_Visualize_Workbench_Externalizing_RandomCV.m, Step_5th_Weight_Visualize_Workbench_Mood_RandomCV.m:
   > Visualize the contribution pattern using workbench.
6. Step_6th_SumofWeights_Fear.R, Step_6th_SumofWeights_Psychosis.R, Step_6th_SumofWeights_Externalizing.R, Step_6th_SumofWeights_Mood.R:
   > Calculate the sum of weights within each of 17 networks and visualize using bar plot. 
7. Step_7th_SumofWeights_Correlation.R:
   > Calculate the correlation between the sum of weights and the median network variability, and then visualizing using scatter plot.
   
### Step_6th_PLSr1_OverallPsyFactor
Prediction of overall psychopathology factor using personalized functional topography. The computations from step1 to step 6 are the same with the above. 

### Step_7th_PLSca
Relate the functional topography and item-level psychopathology symptoms using partial least square correlation.
1. Step_1st_PLSCorr.py:
   > Relate functional topography and item-level psychopathology symptoms using partial least square correlation. Repeated (i.e., 101 times) 2F-CV was used and a permutation testing was used to generate the distribution of accuries at chance level.
2. Step_3rd_CovarianceExplained.R:
   > Calculate the median covariance explained by each component across the 101 times. 
3. Step_5th_Scatter_PLSca.R:
   > The scatter plot of the correlation between the first pair of components. 
4. Step_2nd_PLSca_Sig.m:
   > Calculate the P value of the out-of-sample correlation between pairs of components. 
5. Step_4th_Prediction_RandomCV_All_Violin.R:
   > The plot of the distribution of correlation between first pair of components from permutation testing. 
6. Step_6th_BehaviorBrainFeatures_Update.m:
   > 
7. Step_7th_BrainFeatures.m:
8. Step_8th_SumofWeights_PLSca.R:



# Linking Individual Differences in Personalized Functional Network Topography to Psychopathology in Youth

*The spatial distribution of large-scale functional networks across association cortex is refined in neurodevelopment and is related to individual differences in cognition. However, it remains unknown if individual variability in this functional topography is associated with major dimensions of psychopathology in youth. Capitalizing on advances in machine learning and a large sample imaged with 27 min of high-quality functional MRI (fMRI) data (n = 790, ages 8-23 years), we examined the association between the functional topography and dimensions of psychopathology in youth. We found the topographies of individualized functional networks significantly predicted the four correlated dimensions of psychopathology, including fear, psychosis, externalizing and anxious-misery. The contribution patterns of the four dimensions were similar: the .  Further analysis revealed that individualized functional topography predicted the general psychopathology factor (r = 0.15, p(perm)<0.001), which underlies the significant predictions of the four correlated dimensions. There results provide novel evidence that individual differences in person-specific functional topography in association networks are linked to dimensions of psychopathology in youth. 

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
1. Step_1st_SubjectFilter.R: Selecting our sample (i.e., 790 subjects)
   > Inclusion criteria: healthExcludev2 = 0; fsFinalExclude = 0; restExclude = 0; restExcludeVoxelwise = 0; nbackFcExclude = 0; nbackFcExcludeVoxelwise = 0; idemoFcExclude = 0; idemoFcExcludeVoxelwise = 0. Finally, a sample of 790 subjects was created.
2. Step_2nd_ExtractBehavior.R
   > Extracting correlated dimensions, bifactors, and item-level symptom of psychopathology
3. Step_3rd_CopyStructFSFiles.R, Step_4th_DataFSProcessing.m, Step_5th_MergeModalities.m: 
   > Projecting fmri data into surface and then combine the three modalities
4. Step_6th_tSNRMask*: 
   > Generating the tSNR mask
5. Step_7th_EFA.R, Step_8th_CFA: 
   > Plot figures for exploratory factor analysis (Figure 2) and confirmatory factor analysis (Figure 5)
6. Step_9th_Prevelance.R: 
   > Calculating the prevelance of disorders.
<br>

### Step_2nd_SingleParcellation
Step1 to step4 are codes for single functional parcellation (Li et al., 2017, NeuroImage). See (https://github.com/hmlicas/Collaborative_Brain_Decomposition) for the toolbox of single parcellation.
1. Step_1st_CreatePrepData.m: 
   > Creating the spatial neighborhood for fsaverage5 surface space. After removing medial wall, we have 18715 vertices.
2. Step_2nd_ParcellationInitialize.m:
   > Calculating the group parcellation, which will be the initialization of single subject parcellation. We randomly chose 100 subjects and combined these subjects' data along time points and run non-negative matrix factorization (NMF) to decompose the whole brain into 17 networks. We repeated this procedure 50 times, finally 50 group atlas was acquired.
3. Step_3rd_SelRobustInit.m:
   > Using normalized cuts based spectrum clustering method to cluster the acquired 50 group atlases. Finally, one group atlas with 17 networks was acquired.
4. Step_4th_IndividualParcel.m:
   > Based on the acquired group atlas and the subject's specific fMRI data, we calculated the atlas for this specific subject. Finally, each subject had a loading matrix, in which the loading value quantifies the probability each vertex belonging to each network.
5. Step_5th_AtlasInformation_Extract.m:
   > Extract all subjects' loading matrix information into the same folder. Also, extracting the discrete atlas for each subject by categorizing each vertex to the network with the highest loading, and put them into the same folder.
6. Step_6th_GroupAtlas_Extract.m:
   > Extracting loading matrix group atlas (the output of the script 'Step_3rd_SelRobustInit.m') and create the group discrete atlas by categorizing each vertex to the network with the highest loading.
7. Step_7th_NetworkNaming_Yeo.m:
   > Calculating the overlap between our networks with Yeo atlas.
8. Step_8th_Visualize_Workbench_Atlas.m:
   > Calculating inter-subjects variability of loadings according to the formula: MADM=median(|x(i) - median(x)|). Calculating the average of 17 variability brain map.
9. Step_9th_Visualize_Workbench_AtlasVariability.m:
   > Visualizing the atlas variability map using workbench
10. Step_10th_ViolinPlot_AtlasVariability_Loading.R:
   > Visualizing the variability of each network using violin plot.
<br>

### Step_3rd_Psychopathology_SaveMat.R
   > Saving age, sex, motion, four correlated dimensions of psychopathology, five bifactors of psychopathology, and 112 item-level psychopathology symptoms to .mat file.
   
### Step_4th_AtlasFeature_SaveMat.m
   > For each participant, extracting all the loadings of the 17 networks to be a high-dimensional feature vector. 
   
### Step_5th_PLSr1_Corrtraits
1. 
   
   


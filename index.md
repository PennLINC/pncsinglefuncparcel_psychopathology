Run the codes according to the order:
Step_1st_PrepareData:
Selecting our sample (i.e., 693 subjects).
Inclusion criteria: healthExcludev2 = 0; fsFinalExclude = 0; restExclude = 0; restExcludeVoxelwise = 0; nbackFcExclude = 0; nbackFcExcludeVoxelwise = 0; idemoFcExclude = 0; idemoFcExcludeVoxelwise = 0. Finally, a sample of 790 subjects was created.
Then, doing data pre-processing and doing plot for factor analysis (i.e., exploratory and confirmatory).

Step_2nd_SingleParcellation:
Here, codes for single brain functional parcellation (Li et al., 2017, NeuroImage). See (https://github.com/hmlicas/Collaborative_Brain_Decomposition) for the codes of single parcellation. We parcellated each subject’s brain into 17 networks as did in our prior work (Cui et al., 2020, Neuron).

Step_3rd_AgeCognitionPsycho_SaveMat.R:
Extracting the demographics, factor scores, corrtraits scores and psychopathology items.

Step_4th_AtlasFeature_SaveMat.m:
Extracting the functional topography matrix in the same subjects’ order as scores acquired in Step 3. 

Step_5th_PLSr1_Corrtraits:
Using partial least square regression to predict fear, psychosis, externalizing and mood corrtraits using functional topography. 
Then, visualizing the brain map of prediction weight using workbench, and visualizing the sum of weight in each network using bar graph.

Step_6th_PLSr1_OverallPsyFactor:
Using partial least square regression to predict overall psychopathology factor using functional topography and then visualizing the contribution weight.

Step_7th_PLSca:
Using partial least square correlation to relate the multivariate psychopathology and multivariate functional topography to validate our main results.

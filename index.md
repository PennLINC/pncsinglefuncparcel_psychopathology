
<br>
<br>
# Linking Individual Differences in Personalized Functional Network Topography to Psychopathology in Youth
*The spatial distribution of large-scale functional networks across association cortex is refined in neurodevelopment and is related to individual differences in cognition. However, it remains unknown if individual variability in this functional topography is associated with major dimensions of psychopathology in youth. Capitalizing on advances in machine learning and a large sample imaged with 27 min of high-quality functional MRI (fMRI) data (n = 790, ages 8-23 years), we examined the association between the functional topography and dimensions of psychopathology in youth. We found the topographies of individualized functional networks significantly predicted the four correlated dimensions of psychopathology, including fear, psychosis, externalizing and anxious-misery. The contribution patterns of the four dimensions were similar: the .  Further analysis revealed that individualized functional topography predicted the general psychopathology factor (r = 0.15, p(perm)<0.001), which underlies the significant predictions of the four correlated dimensions. There results provide novel evidence that individual differences in person-specific functional topography in association networks are linked to dimensions of psychopathology in youth. 

### Project Lead
Zaixu Cui

### Faculty Leads
Theodore D. Satterthwaite

### Analytic Replicator
Adam Pines

### Collaborators 
Adam R. Pines, Hongming Li, Tyler M. Moore, Azeez Adebimpe, Jacob W. Vogel, Sheila Shanmugan, Bart Larsen, Max Bertolero, Cedric H. Xia, Raquel E. Gur, Ruben C. Gur, Desmond J. Oathes, Aaron F. Alexander-Bloch, Michael P. Milham, Giovanni A. Salum, Monica E. Calkins, David R. Roalf, Russell T. Shinohara, Daniel H. Wolf, Christos Davatzikos, Danielle S. Bassett, Damien A. Fair, Yong Fan, and Theodore D. Satterthwaite

### Current Project Status
In preparation.

### Datasets
PNC dataset.

### Github Repository
<https://github.com/PennLINC/pncsinglefuncparcel_psychopathology>

### Path to Data on Filesystem 
/data/jux/BBL/projects/sydnor_glucest/GluCEST_BASReward_Project







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

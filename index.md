
<br>
<br>
# Linking Individual Differences in Personalized Functional Network Topography to Psychopathology in Youth
*The spatial distribution of large-scale functional networks across association cortex is refined in neurodevelopment and is related to individual differences in cognition. However, it remains unknown if individual variability in this functional topography is associated with major dimensions of psychopathology in youth. Capitalizing on advances in machine learning and a large sample imaged with 27 min of high-quality functional MRI (fMRI) data (n = 790, ages 8-23 years), we examined the association between the functional topography and dimensions of psychopathology in youth. We found the topographies of individualized functional networks significantly predicted the four correlated dimensions of psychopathology, including fear, psychosis, externalizing and anxious-misery. 
Similar association networks, including fronto-parietal, ventral attention and dorsal attention, represented the highest negative contribution weight for the prediction of these 

Glutamate-modulating psychotropics such as ketamine have recently shown efficacy for treating anhedonia, motivating interest in parsing relationships between glutamate and reward functioning. Moreover, rodent and non-human primate studies have demonstrated that antagonism of AMPA and group 1 metabotropic glutamate receptors decreases reward sensitivity, whereas optogenetic activation of reward network glutamatergic afferents is reinforcing. Despite this convergent evidence from clinical and preclinical studies suggesting that glutamate availability may modulate reward responsiveness, validation of such a relationship in humans has been difficult, given limitations in our ability to study glutamate in the human brain. In this project, we therefore capitalize on the novel, ultra-high field imaging method GluCEST (Glutamate Chemical Exchange Saturation Transfer), which offers enhanced sensitivity, spatial coverage, and spatial resolution for imaging glutamate in vivo. Specifically, this project capitalizes on GluCEST data collected at 7T from a transdiagnostic population (healthy, depression, psychosis-spectrum) to test the hypothesis that lower levels of glutamate within the brain's reward network are dimensionally associated with diminished reward responsiveness. Given that diminished reward responsiveness is linked to poor psychological wellbeing, psychiatric disorder risk, suicidal ideation, and psychotropic treatment resistance, understanding the neural mechanisms underlying this clinical phenotype is vitally important.*


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

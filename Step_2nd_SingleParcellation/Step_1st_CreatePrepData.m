
clear
ProjectFolder = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho/results/SingleParcellation';
mkdir(ProjectFolder);

SubjectsFolder = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho/freesurfer/6.0.0/subjects/fsaverage5';
% for surface data
surfL = [SubjectsFolder '/surf/lh.pial'];
surfR = [SubjectsFolder '/surf/rh.pial'];
surfML = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho/data/SNR_Mask/subjects/fsaverage5/lh.Mask_SNR.label';
surfMR = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho/data/SNR_Mask/subjects/fsaverage5/rh.Mask_SNR.label';

[surfStru, surfMask] = getFsSurf(surfL, surfR, surfML, surfMR);

gNb = createPrepData('surface', surfStru, 1, surfMask);

% save gNb into file for later use
prepDataName = [ProjectFolder '/CreatePrepData.mat'];
save(prepDataName, 'gNb');


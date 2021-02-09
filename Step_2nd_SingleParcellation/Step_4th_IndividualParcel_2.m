
%
% Based on the group atlas, creating each subject's individual specific atlas
% For the toolbox of single brain parcellation, see: 
%

clear

ProjectFolder = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho';
SingleParcellationFolder = [ProjectFolder '/results/SingleParcellation'];
ResultantFolder = [SingleParcellationFolder '/SingleParcel_1by1'];
mkdir(ResultantFolder);

PrepDataFile = [SingleParcellationFolder '/CreatePrepData.mat'];
resId = 'IndividualParcel_Final';
initName = [SingleParcellationFolder '/RobustInitialization/init.mat'];
K = 17;
% Use parameter in Hongming's NeuroImage paper
alphaS21 = 1;
alphaL = 10;
vxI = 1;
spaR = 1;
ard = 0;
iterNum = 30;
eta = 0;
calcGrp = 0;
parforOn = 0;

SubjectsFolder = [ProjectFolder '/freesurfer/6.0.0/subjects/fsaverage5'];
% for surface data
surfML = [ProjectFolder '/data/SNR_Mask/subjects/fsaverage5/lh.Mask_SNR.label'];
surfMR = [ProjectFolder '/data/SNR_Mask/subjects/fsaverage5/rh.Mask_SNR.label'];

RawDataFolder = [ProjectFolder '/data/SurfaceData/CombinedData'];
LeftCell = g_ls([RawDataFolder '/*/lh.fs5.sm6.residualised.mgh']);
RightCell = g_ls([RawDataFolder '/*/rh.fs5.sm6.residualised.mgh']);

% Parcellate for each subject separately
for i = 1:length(LeftCell)
    i
    [Fold, ~, ~] = fileparts(LeftCell{i});
    [~, ID_Str, ~] = fileparts(Fold);
    ID = str2num(ID_Str);
    ResultantFolder_I = [ResultantFolder '/Sub_' ID_Str];
    ResultantFile = [ResultantFolder_I '/IndividualParcel_Final_sbj1_comp17_alphaS21_1_alphaL10_vxInfo1_ard0_eta0/final_UV.mat'];
    if ~exist(ResultantFile, 'file');
        mkdir(ResultantFolder_I);
        IDMatFile = [ResultantFolder_I '/ID.mat'];
        save(IDMatFile, 'ID');

        sbjListFile = [ResultantFolder_I '/sbjListAllFile_' num2str(i) '.txt'];
        system(['rm ' sbjListFile]);

        cmd = ['echo ' LeftCell{i} ' >> ' sbjListFile];
        system(cmd);
        cmd = ['echo ' RightCell{i} ' >> ' sbjListFile];
        system(cmd);

        deployFuncMvnmfL21p1_func_surf_fs(sbjListFile,surfML,surfMR, ...
          PrepDataFile,ResultantFolder_I,resId,initName,K,alphaS21, ...
          alphaL,vxI,spaR,ard,eta,iterNum,calcGrp,parforOn);
        pause(1);
    end
end



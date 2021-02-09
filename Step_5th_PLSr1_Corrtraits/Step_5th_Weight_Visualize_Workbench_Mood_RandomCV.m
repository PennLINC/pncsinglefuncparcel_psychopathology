
clear
ProjectFolder = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho';
PredictionFolder = [ProjectFolder '/results/PLSr1/AtlasLoading/'];
Results_Cell = g_ls([PredictionFolder '/MoodCorrtraits_All_RegressCovariates_RandomCV/Time_*/Fold_*_Score.mat']);
for i = 1:length(Results_Cell)
  tmp = load(Results_Cell{i});
  w_Brain_Mood_AllModels(i, :) = tmp.w_Brain;
end
w_Brain_Mood = median(w_Brain_Mood_AllModels);
VisualizeFolder = [PredictionFolder '/WeightVisualize_Mood_RandomCV'];
mkdir(VisualizeFolder);
save([VisualizeFolder '/w_Brain_Mood.mat'], 'w_Brain_Mood');

% for surface data
surfML = [ProjectFolder '/data/SNR_Mask/subjects/fsaverage5/lh.Mask_SNR.label'];
surfMR = [ProjectFolder '/data/SNR_Mask/subjects/fsaverage5/rh.Mask_SNR.label'];
mwIndVec_l = read_medial_wall_label(surfML);
Index_l = setdiff([1:10242], mwIndVec_l);
mwIndVec_r = read_medial_wall_label(surfMR);
Index_r = setdiff([1:10242], mwIndVec_r);

load([ProjectFolder '/results/AtlasData/AtlasLoading/AtlasLoading_All_RemoveZero.mat']); % NonZeroIndex was here
%%%%%%%%%%%%%%%%%%
% Mood Prediction %
%%%%%%%%%%%%%%%%%%
VertexQuantity = 17754;
%% Display sum absolute weight of the 17 maps
w_Brain_Mood_All = zeros(1, 17754*17);
w_Brain_Mood_All(NonZeroIndex) = w_Brain_Mood;
%% Display weight of all regions
for i = 1:17
    w_Brain_Mood_Matrix(i, :) = w_Brain_Mood_All([(i - 1) * VertexQuantity + 1 : i * VertexQuantity]);
end
save([VisualizeFolder '/w_Brain_Mood_Matrix.mat'], 'w_Brain_Mood_Matrix');

w_Brain_Mood_Abs_sum = sum(abs(w_Brain_Mood_Matrix));
w_Brain_Mood_Abs_sum_lh = zeros(1, 10242);
w_Brain_Mood_Abs_sum_lh(Index_l) = w_Brain_Mood_Abs_sum(1:length(Index_l));
w_Brain_Mood_Abs_sum_rh = zeros(1, 10242);
w_Brain_Mood_Abs_sum_rh(Index_r) = w_Brain_Mood_Abs_sum(length(Index_l) + 1:end);
save([VisualizeFolder '/w_Brain_Mood_Abs_sum.mat'], 'w_Brain_Mood_Abs_sum', ...
                         'w_Brain_Mood_Abs_sum_lh', 'w_Brain_Mood_Abs_sum_rh');

V_lh = gifti;
V_lh.cdata = w_Brain_Mood_Abs_sum_lh';
V_lh_File = [VisualizeFolder '/w_Brain_Mood_Abs_sum_RandomCV_lh.func.gii'];
save(V_lh, V_lh_File);
pause(1);
V_rh = gifti;
V_rh.cdata = w_Brain_Mood_Abs_sum_rh';
V_rh_File = [VisualizeFolder '/w_Brain_Mood_Abs_sum_RandomCV_rh.func.gii'];
save(V_rh, V_rh_File);
% combine 
cmd = ['wb_command -cifti-create-dense-scalar ' VisualizeFolder '/w_Brain_Mood_Abs_sum_RandomCV' ...
         '.dscalar.nii -left-metric ' V_lh_File ' -right-metric ' V_rh_File];
system(cmd);
pause(1);
system(['rm -rf ' V_lh_File ' ' V_rh_File]);


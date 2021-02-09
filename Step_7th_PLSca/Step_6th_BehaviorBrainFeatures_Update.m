
clear
ProjectFolder = '/cbica/projects/GURLAB/projects/pncSingleFuncParcel_psycho';
PLSca_Folder = [ProjectFolder '/results/PLSca/AtlasLoading'];
Res_Cell = g_ls([PLSca_Folder '/RandomCV_101Repeats_RegressCovariates_All_2Fold/*/Fold_*_Score.mat']);
% Correct the sign to make most signs in behavior side is positive
% Because we have 112 behavior feaature
for i = 1:length(Res_Cell)
  i
  tmp = load(Res_Cell{i});
  if length(find(tmp.Behavior_Weight(:, 1) > 0)) > 56 
    Behavior_Weight_New(i, :) = tmp.Behavior_Weight(:, 1);    
    Brain_Weight_New(i, :) = tmp.Brain_Weight(:, 1);
  else 
    Behavior_Weight_New(i, :) = -tmp.Behavior_Weight(:, 1);
    Brain_Weight_New(i, :) = -tmp.Brain_Weight(:, 1);    
  end
end

% Calculating BSR
BSR = median(Behavior_Weight_New) ./ std(Behavior_Weight_New);
length(find(BSR > 2.576)) % 2.576 corresponds to P=0.01

save([PLSca_Folder '/RandomCV_101Repeats_RegressCovariates_All_2Fold/Weight_AllSubjects_Update.mat'], 'Behavior_Weight_New', 'Brain_Weight_New');


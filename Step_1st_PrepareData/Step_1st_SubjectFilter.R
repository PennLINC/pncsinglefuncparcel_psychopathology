##########################################################################################################
### This script will construct a subject sample for PNC-SingleSubjParcellation-Development analyses, retaining  ###
###     subjects who passed quality assurance protocols for Freesurfer, healthExcludev2, resting, nback, emotion fMRI.      ###
###                                 Finally, 790 subjects remained.                                    ###
##########################################################################################################

library(R.matlab)

#########################################################
### 1. Filter subjects, finally 790 subjects remained ###
#########################################################
DemograBehaviorFolder = '/cbica/projects/pncSingleFuncParcel/pncSingleFuncParcel_psycho/data/processedData/DemograBehavior';
# Health Status
health <- read.csv(paste0(DemograBehaviorFolder, "/n1601_health_20170421.csv"))
# T1
t1_qa <- read.csv(paste0(DemograBehaviorFolder, "/n1601_t1QaData_20170306.csv"))
# REST
REST_qa <- read.csv(paste0(DemograBehaviorFolder, "/n1601_RestQAData_20170714.csv"));
# nback
nback_qa <- read.csv(paste0(DemograBehaviorFolder, "/n1601_NbackConnectTaskRegressQAData_2018-10-21.csv"));
# idemo (emotion identification)
idemo_qa <- read.csv(paste0(DemograBehaviorFolder, "/n1601_IdemoConnectQAData_20170718.csv"));
# Merge QA data 
df <- health
df <- merge(df, t1_qa, by=c("scanid","bblid"))
df <- merge(df, REST_qa, by=c("scanid","bblid"))
df <- merge(df, nback_qa, by=c("scanid","bblid"))
df <- merge(df, idemo_qa, by=c("scanid","bblid"))
## Define study sample using exclusion criteria
# health Exclude
df <- df[which(df$healthExcludev2 == 0), ]
# FreeSurfer QA
df <- df[which(df$fsFinalExclude == 0), ] 
# REST QA
df <- df[which(df$restExclude==0 & df$restExcludeVoxelwise==0), ]
# nback QA
df <- df[which(df$nbackFcExclude==0 & df$nbackFcExcludeVoxelwise==0), ]
# idemo QA
df <- df[which(df$idemoFcExclude==0 & df$idemoFcExcludeVoxelwise==0), ]
####################################
# After above criteria, 790 subjects remaining.
###################################
# Specify columns to retain
attach(df)
keeps <- c("bblid", "scanid")
# Define new dataframe with specified columns 
subjid_df <- df[keeps]
dim(subjid_df)
detach(df)
# Write out subject identifiers to CSV
# Finally, we get 790 subjects
write.csv(subjid_df, "/cbica/projects/pncSingleFuncParcel/pncSingleFuncParcel_psycho/data/pncSingleFuncParcelPsychopatho_n790_SubjectsIDs.csv", row.names=FALSE)


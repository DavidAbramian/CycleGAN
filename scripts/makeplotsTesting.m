close all
clear all
clc

addpath('/home/andek67/Research_projects/nifti_matlab')

rotateData = 0;

study = 'Beijing';
%study = 'Cambridge';

numberOfTestSubjects = 38;
subjectOffset = 160;

if strcmp(study,'Beijing')
    
    synthpaths{1} = 'runs/20190906-160147-fcon1000_64_Beijing_augmented_LR_0.0002_RL_6_DF_64_GF_32_RF_46_augmented_rot/synthetic_volumes/fcon1000_64_Beijing_augmented/';
    
    realpath = 'data/fcon1000_64_Beijing_augmented/';
else
    
    realpath = 'data/fcon1000_64_Cambridge/';
end

epochs = 5:5:30;

legendText{1} = 'LR 2e-4 RL 6 DF 64 GF 32 RF 46 augmented rotations';

correlationMeansT1 = NaN*zeros(length(epochs),length(synthpaths));
correlationStdsT1 = NaN*zeros(length(epochs),length(synthpaths));
allCorrelationsT1 = NaN*zeros(length(epochs),length(synthpaths),numberOfTestSubjects);

mseMeansT1 = NaN*zeros(length(epochs),length(synthpaths));
allMsesT1 = NaN*zeros(length(epochs),length(synthpaths),numberOfTestSubjects);

realfiles = dir([realpath 'testB/']);

synthfiles = realfiles;

% Create synthetic file names from real file names, by replacing T1 with
% fMRI, and adding 'synthetic'
for subject = 1:numberOfTestSubjects
   synthfiles(subject+2).name = strrep(synthfiles(subject+2).name,'T1','fMRI');
   synthfiles(subject+2).name = strrep(synthfiles(subject+2).name,'.nii.gz','_synthetic.nii.gz');
end


for setting = 1:length(synthpaths)
     
    setting
    
    epochIteration = 1;
     
    for epoch = epochs
        
        correlations = zeros(numberOfTestSubjects,1);
        mses = zeros(numberOfTestSubjects,1);
        
        for subject = 1:numberOfTestSubjects
            
            % Load real volume
            
            %nii = load_nii([realpath 'testB/' study '_T1_' num2str(subjectOffset+subject) '.nii.gz']);
            nii = load_nii([realpath 'testB/' realfiles(subject+2).name ]);
            T1real = double(nii.img);
            
            try
                
                % Load synthetic volume
                
                %nii = load_nii([synthpaths{setting} 'epoch_' num2str(epoch) '/A2B/' study '_fMRI_' num2str(subjectOffset+subject) '_synthetic.nii.gz']);
                nii = load_nii([synthpaths{setting} 'epoch_' num2str(epoch) '/A2B/' synthfiles(subject+2).name ]);
                filetoload = synthfiles(subject+2).name;
                T1synth = double(nii.img);
                [sy sx sz] = size(T1synth);
                
                if rotateData == 1
                    for z = 1:sz
                        T1synth(:,:,z) = flipud(T1synth(:,:,z)); 
                    end
                end
                
                mask = (T1real > 20);
                T1real = T1real(mask);
                T1synth = T1synth(mask);
                correlations(subject) = corr2(T1real(:),T1synth(:));
                mses(subject) = mean( (T1real(:) - T1synth(:)).^2 );
                allCorrelationsT1(epochIteration,setting,subject) = correlations(subject);
                allMsesT1(epochIteration,setting,subject) = mses(subject);
                
            catch
                disp('Could not find data') 
                synthfiles(subject+2).name
            end
               
        end
        
        correlationMeansT1(epochIteration,setting) = mean(correlations);
        correlationStdsT1(epochIteration,setting) = std(correlations);
        
        mseMeansT1(epochIteration,setting) = mean(mses);
        
        epochIteration = epochIteration + 1;
        
    end
    
end

figure
plot(epochs,correlationMeansT1)
legend(legendText)
title('T1 real T1 synth correlation mean')
axis([0 epochs(end) 0.6 1.0])

figure
plot(epochs,mseMeansT1)
legend(legendText)
title('T1 real T1 synth mse mean')

%----

correlationMeansfMRI = NaN*zeros(length(epochs),length(synthpaths));
correlationStdsfMRI = NaN*zeros(length(epochs),length(synthpaths));
allCorrelationsfMRI = NaN*zeros(length(epochs),length(synthpaths),numberOfTestSubjects);

mseMeansfMRI = NaN*zeros(length(epochs),length(synthpaths));
allMsesfMRI = NaN*zeros(length(epochs),length(synthpaths),numberOfTestSubjects);

realfiles = dir([realpath 'testA/']);

synthfiles = realfiles;

% Create synthetic file names from real file names, by replacing T1 with
% fMRI, and adding 'synthetic'
for subject = 1:numberOfTestSubjects
   synthfiles(subject+2).name = strrep(synthfiles(subject+2).name,'fMRI','T1');
   synthfiles(subject+2).name = strrep(synthfiles(subject+2).name,'.nii.gz','_synthetic.nii.gz');
end

for setting = 1:length(synthpaths)
    
    setting
    
    epochIteration = 1;
    
    for epoch = epochs
        
        correlations = zeros(numberOfTestSubjects,1);
        mses = zeros(numberOfTestSubjects,1);
        
        for subject = 1:numberOfTestSubjects
            
            % Load real volume
            
            %nii = load_nii([realpath 'testA/' study '_fMRI_' num2str(subjectOffset+subject) '.nii.gz']);
            nii = load_nii([realpath 'testA/' realfiles(subject+2).name ]);
            fMRIreal = double(nii.img);
            
            try
                % Load synthetic volume
                
                %nii = load_nii([synthpaths{setting} 'epoch_' num2str(epoch) '/B2A/' study '_T1_' num2str(subjectOffset+subject) '_synthetic.nii.gz']);
                nii = load_nii([synthpaths{setting} 'epoch_' num2str(epoch) '/B2A/' synthfiles(subject+2).name ]);
                fMRIsynth = double(nii.img);
                [sy sx sz] = size(fMRIsynth);
                
                if rotateData == 1
                    for z = 1:sz
                        fMRIsynth(:,:,z) = flipud(fMRIsynth(:,:,z)); 
                    end
                end
                
                mask = (fMRIreal > 20);
                fMRIreal = fMRIreal(mask);
                fMRIsynth = fMRIsynth(mask);
                correlations(subject) = corr2(fMRIreal(:),fMRIsynth(:));
                mses(subject) = mean( (fMRIreal(:) - fMRIsynth(:)).^2 );
                allCorrelationsfMRI(epochIteration,setting,subject) = correlations(subject);
                allMsesfMRI(epochIteration,setting,subject) = mses(subject); 
            catch
                disp('Could not find data')
            end
            
        end
        
        correlationMeansfMRI(epochIteration,setting) = mean(correlations);
        correlationStdsfMRI(epochIteration,setting) = std(correlations);
        
        mseMeansfMRI(epochIteration,setting) = mean(mses);
        
        epochIteration = epochIteration + 1;
        
    end
    
end

figure
plot(epochs,correlationMeansfMRI)
legend(legendText)
title('fMRI real fMRI synth correlation mean')
axis([0 epochs(end) 0.75 1.0])

figure
plot(epochs,mseMeansfMRI)
legend(legendText)
title('fMRI real fMRI synth mse mean')




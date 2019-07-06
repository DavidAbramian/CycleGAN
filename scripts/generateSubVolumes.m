close all
clear all
clc

addpath('/home/andek67/Research_projects/nifti_matlab')

cubesize = 48; % factor of 4

HCPDirectory='/flush2/andek67/Data/HCP/STRUCTURAL/';
outputDirectory='/home/andek67/Research_projects/CycleGAN3D/data/HCP_T1T2_48cubes';

numberOfSubjects = 1113
numberOfTrainingSubjects = 900;
numberOfSubvolumesPerSubject = 10;

directories = dir(HCPDirectory);

subvolumeNumber = 1;

for subject = 1:numberOfSubjects

	subject

	if subject == (numberOfTrainingSubjects + 1)
		subvolumeNumber = 1;
	end

	subjectname = directories(subject+2).name

    % Load full nifti volumes
    nii = load_nii([HCPDirectory subjectname '/MNINonLinear/T1w_restore_brain.nii.gz']);
    T1 = single(nii.img);

    nii = load_nii([HCPDirectory subjectname '/MNINonLinear/T2w_restore_brain.nii.gz']);
    T2 = single(nii.img);

    [sy,sx,sz] = size(T1);

    for v = 1:numberOfSubvolumesPerSubject

        % Randomise x,y,z coordinates
        x = randi(sx - cubesize*2,1) + cubesize;
        y = randi(sy - cubesize*2,1) + cubesize;
        z = randi(sz - cubesize*2,1) + cubesize;

		% Get 32 cube subvolumes
        subvolumeT1 = T1( (y-cubesize/2):(y+(cubesize/2-1)) , (x-cubesize/2):(x+(cubesize/2-1)), (z-cubesize/2):(z+(cubesize/2-1)));
		smallT1 = make_nii(subvolumeT1, [0.7 0.7 0.7], [0 0 0], 16);

        subvolumeT2 = T2( (y-cubesize/2):(y+(cubesize/2-1)) , (x-cubesize/2):(x+(cubesize/2-1)), (z-cubesize/2):(z+(cubesize/2-1)));
		smallT2 = make_nii(subvolumeT2, [0.7 0.7 0.7], [0 0 0], 16);

		if subject <= numberOfTrainingSubjects
			save_nii(smallT1,[outputDirectory '/trainA/T1_' num2str(subvolumeNumber) '.nii.gz' ]);
			save_nii(smallT2,[outputDirectory '/trainB/T2_' num2str(subvolumeNumber) '.nii.gz' ]);
		else
			save_nii(smallT1,[outputDirectory '/testA/T1_' num2str(subvolumeNumber) '.nii.gz' ]);
			save_nii(smallT2,[outputDirectory '/testB/T2_' num2str(subvolumeNumber) '.nii.gz' ]);
		end

        subvolumeNumber = subvolumeNumber + 1;
    end
end


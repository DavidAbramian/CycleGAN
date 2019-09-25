close all
clear all
clc

addpath('/home/andek67/Research_projects/nifti_matlab')

cubesize = 32; % factor of 4
cubesizeZ = 32;

dataDirectory='/home/andek67/Research_projects/CycleGAN3D/data/fcon1000_128_Beijing/';
outputDirectory='/home/andek67/Research_projects/CycleGAN3D/data/fcon1000_32cubeshighres_Beijing';

numberOfSubjects = 198
numberOfTrainingSubjects = 160;
numberOfSubvolumesPerSubject = 50;

voxelSize = 1;

subvolumeNumber = 1;

for subject = 1:numberOfTrainingSubjects

	subject

    % Load full nifti volumes
    nii = load_nii([dataDirectory '/trainA/Beijing_fMRI_' num2str(subject) '.nii.gz']);
    fMRI = single(nii.img);

    nii = load_nii([dataDirectory '/trainB/Beijing_T1_' num2str(subject) '.nii.gz' ]);
    T1 = single(nii.img);
    
    [sy,sx,sz] = size(T1);

    v = 1;
    while  (v <= numberOfSubvolumesPerSubject)

        % Randomise x,y,z coordinates
        x = max(min( randi(sx,1,1),sx-cubesize/2),cubesize/2+1);
        y = max(min( randi(sy,1,1),sy-cubesize/2),cubesize/2+1);
        z = max(min( randi(sz,1,1),sz-cubesizeZ/2),cubesizeZ/2+1);

		% Get 32 cube subvolumes
        subvolumefMRI = fMRI( (y-cubesize/2):(y+(cubesize/2-1)) , (x-cubesize/2):(x+(cubesize/2-1)), (z-cubesizeZ/2):(z+(cubesizeZ/2-1)));
		smallfMRI = make_nii(subvolumefMRI, [voxelSize voxelSize voxelSize], [0 0 0], 16);
        
        subvolumeT1 = T1( (y-cubesize/2):(y+(cubesize/2-1)) , (x-cubesize/2):(x+(cubesize/2-1)), (z-cubesizeZ/2):(z+(cubesizeZ/2-1)));
		smallT1 = make_nii(subvolumeT1, [voxelSize voxelSize voxelSize], [0 0 0], 16);

        proportionZeros = sum(subvolumefMRI(:) == 0) / (cubesize * cubesize * cubesizeZ);
        
		if (proportionZeros < 0.7)
            
			save_nii(smallfMRI,[outputDirectory '/trainA/Beijing_fMRI_' num2str(subvolumeNumber) '.nii.gz' ]);
			save_nii(smallT1,[outputDirectory '/trainB/Beijing_T1_' num2str(subvolumeNumber) '.nii.gz' ]);
            v = v + 1;
            subvolumeNumber = subvolumeNumber + 1;
            
        end

    end
end


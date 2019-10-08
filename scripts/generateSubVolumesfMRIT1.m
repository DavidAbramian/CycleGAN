close all
clear all
clc

addpath('/home/andek67/Research_projects/nifti_matlab')

cubesize = 48; % factor of 4
cubesizeZ = 48;

dataDirectory='/home/andek67/Research_projects/CycleGAN3D/data/fcon1000_64_Beijing_augmented20/';
outputDirectory='/home/andek67/Research_projects/CycleGAN3D/data/fcon1000_48cubes_Beijing_augmented20';

originalDataA = dir([dataDirectory '/trainA']);
originalDataA = originalDataA(3:end);

originalDataB = dir([dataDirectory '/trainB']);
originalDataB = originalDataB(3:end);

numberOfSubvolumesPerSubject = 3;

voxelSize = 2;

subvolumeNumber = 1;

for subject = 1:length(originalDataA)

	subject

    % Load full nifti volumes
    nii = load_nii([dataDirectory '/trainA/' originalDataA(subject).name ]);
    fMRI = single(nii.img);

    nii = load_nii([dataDirectory '/trainB/' originalDataB(subject).name ]);
    T1 = single(nii.img);
    
    [sy,sx,sz] = size(T1);

    v = 1;
    while  (v <= numberOfSubvolumesPerSubject)

        % Randomise x,y,z coordinates
        x = max(min( randi(sx,1,1),sx-cubesize/2),cubesize/2+1);
        y = max(min( randi(sy,1,1),sy-cubesize/2),cubesize/2+1);
        z = max(min( randi(sz,1,1),sz-cubesizeZ/2),cubesizeZ/2+1);

		% Get subvolumes
        subvolumefMRI = fMRI( (y-cubesize/2):(y+(cubesize/2-1)) , (x-cubesize/2):(x+(cubesize/2-1)), (z-cubesizeZ/2):(z+(cubesizeZ/2-1)));
		smallfMRI = make_nii(subvolumefMRI, [voxelSize voxelSize voxelSize], [0 0 0], 16);
        
        subvolumeT1 = T1( (y-cubesize/2):(y+(cubesize/2-1)) , (x-cubesize/2):(x+(cubesize/2-1)), (z-cubesizeZ/2):(z+(cubesizeZ/2-1)));
		smallT1 = make_nii(subvolumeT1, [voxelSize voxelSize voxelSize], [0 0 0], 16);

        proportionZeros = sum(subvolumefMRI(:) < 10) / (cubesize * cubesize * cubesizeZ);
        
		if (proportionZeros < 0.5)
            
			save_nii(smallfMRI,[outputDirectory '/trainA/Beijing_fMRI_' num2str(subvolumeNumber) '.nii.gz' ]);
			save_nii(smallT1,[outputDirectory '/trainB/Beijing_T1_' num2str(subvolumeNumber) '.nii.gz' ]);
            v = v + 1;
            subvolumeNumber = subvolumeNumber + 1;
            
        end

    end
end


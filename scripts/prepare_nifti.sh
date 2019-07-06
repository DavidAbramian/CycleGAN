#!/bin/bash

startDirectory=/flush2/andek67/Data/HCP/STRUCTURAL/
dataDirectory=/home/andek67/Research_projects/CycleGAN3D/data/HCP_T1T2

SubjectNumber=1
trainingSubjects=900

for i in ${StartDirectory}* ; do

    echo "This is directory " $i

    # Go to current directory
    cd $i
    # Get subject name
    Subject=${PWD##*/}
    #echo "Processing" $Subject
    # Go back to original directory
    cd $startDirectory

	if [ "$SubjectNumber" -le "$trainingSubjects" ]; then

		echo "Copying" $SubjectNumber to training

		cp ${i}/MNINonLinear/T1w_restore.2.nii.gz ${dataDirectory}/trainA/T1_${SubjectNumber}.nii.gz

		cp ${i}/MNINonLinear/T2w_restore.2.nii.gz ${dataDirectory}/trainB/T2_${SubjectNumber}.nii.gz

		flirt -in ${dataDirectory}/trainA/T1_${SubjectNumber}.nii.gz -ref ${dataDirectory}/dummy_92_112_92.nii.gz -applyxfm -init id.mtx -out ${dataDirectory}/trainA/T1_${SubjectNumber}.nii.gz

		flirt -in ${dataDirectory}/trainB/T2_${SubjectNumber}.nii.gz -ref ${dataDirectory}/dummy_92_112_92.nii.gz -applyxfm -init id.mtx -out ${dataDirectory}/trainB/T2_${SubjectNumber}.nii.gz

	else

		echo "Copying" $SubjectNumber to testing

		cp ${i}/MNINonLinear/T1w_restore.2.nii.gz ${dataDirectory}/testA/T1_${SubjectNumber}.nii.gz

		cp ${i}/MNINonLinear/T2w_restore.2.nii.gz ${dataDirectory}/testB/T2_${SubjectNumber}.nii.gz

		flirt -in ${dataDirectory}/testA/T1_${SubjectNumber}.nii.gz -ref ${dataDirectory}/dummy_92_112_92.nii.gz -applyxfm -init id.mtx -out ${dataDirectory}/testA/T1_${SubjectNumber}.nii.gz

		flirt -in ${dataDirectory}/testB/T2_${SubjectNumber}.nii.gz -ref ${dataDirectory}/dummy_92_112_92.nii.gz -applyxfm -init id.mtx -out ${dataDirectory}/testB/T2_${SubjectNumber}.nii.gz

	fi
	
	((SubjectNumber++))

done




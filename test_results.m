% Compare refaced data with ground truth. Calculate L1 and L2 norms,
% correlation, SSIM. Plot training convergence accross epochs.

currentRun = '20190902-164132-fcon1000_64_Beijing_LR_0.0002_RL_9_DF_64_GF_32_RF_46';
currentDataset = 'fcon1000_64_Beijing';

testA2B = true;

if testA2B
    dirA = fullfile('data', currentDataset,'testA');
    dirB = fullfile('data', currentDataset,'testB');
    dirBSynthString = 'A2B';
else
    dirA = fullfile('data', currentDataset,'testB');
    dirB = fullfile('data', currentDataset,'testA');
    dirBSynthString = 'B2A';
end

dirBSynth = fullfile('runs', currentRun, 'synthetic_volumes', currentDataset);

% Find out saved epochs
epochDirs = dir(fullfile(dirBSynth,'epoch_*'));
epochDirs = {epochDirs.name};
epochsList = sort(str2double(replace(a,'epoch_','')));
nEpochs = length(epochsList);

% Find number of images and size
AList = dir(fullfile(dirA, '*.nii.gz'));
BList = dir(fullfile(dirB, '*.nii.gz'));

nImages = length(AList);

imTest = niftiread(fullfile(AList(1).folder, AList(1).name));
dim = size(imTest);

% Load all ground truth images
AImages = zeros(dim(1), dim(2), dim(3), nImages);
BImages = zeros(dim(1), dim(2), dim(3), nImages);
for i = 1:nImages
    imA = niftiread(fullfile(dirA, AList(i).name));
    AImages(:,:,:,i) = imA;

    imB = niftiread(fullfile(dirB, BList(i).name));
    BImages(:,:,:,i) = imB;
end

%% Calculate L1 and L2 norms of errors and correlation with ground truth 

% Total norm for each epoch
l1normsBBSynth = zeros(nEpochs,1);
l2normsBBSynth = zeros(nEpochs,1);

% Norm for each image and epoch separately
l1normsBBSynthAll = zeros(nEpochs,nImages);
l2normsBBSynthAll = zeros(nEpochs,nImages);

% Correlations
corrBBSynth = zeros(nEpochs,1);
corrBBSynthAll = zeros(nEpochs,nImages);

% Structural similarities (SSIM)
ssimBBSynthAll = zeros(nEpochs,1);

for e = 1:nEpochs
    fprintf('e: %i \n', e)

    epochString = ['epoch_', num2str(epochsList(e))];
    dirBSynthEpoch = fullfile(dirBSynth, epochString, dirBSynthString);

    BSynthList = dir(fullfile(dirBSynthEpoch, '*.nii.gz'));
    BSynthImages = zeros(dim(1), dim(2), dim(3), nImages);

    for i = 1:nImages
        imB = BImages(:,:,:,i);

        imBSynth = double(niftiread(fullfile(BSynthList(i).folder, BSynthList(i).name)));
        BSynthImages(:,:,:,i) = imBSynth;

        l1normsBBSynthAll(e,i) = norm(imB(:) - imBSynth(:), 1);
        l2normsBBSynthAll(e,i) = norm(imB(:) - imBSynth(:), 2);

        corrBBSynthAll(e,i) = corr(imB(:), imBSynth(:));
        ssimBBSynthAll(e,i) = ssim(imB/255, imBSynth/255);
    end

    l1normsBBSynth(e) = norm(BImages(:) - BSynthImages(:), 1);
    l2normsBBSynth(e) = norm(BImages(:) - BSynthImages(:), 2);

    corrBBSynth(e) = corr(BImages(:), BSynthImages(:));
end

% Defaced images
l1normAB = zeros(nEpochs,1);
l1normABAll = zeros(nEpochs,nImages);

l2normAB = zeros(nEpochs,1);
l2normABAll = zeros(nEpochs,nImages);

corrAB = zeros(nEpochs,1);
corrABAll = zeros(nEpochs,nImages);

ssimABAll = zeros(nEpochs,nImages);

for i = 1:nImages
    imA = AImages(:,:,:,i);
    imB = BImages(:,:,:,i);

    l1normABAll(:,i) = norm(imA(:) - imB(:), 1);
    l2normABAll(:,i) = norm(imA(:) - imB(:), 2);

    corrABAll(:,i) = corr(imA(:), imB(:));
    ssimABAll(:,i) = ssim(imA/255, imB/255);
end

l1normAB(:) = norm(AImages(:) - BImages(:), 1);
l2normAB(:) = norm(AImages(:) - BImages(:), 2);

corrAB(:) = corr(AImages(:), BImages(:));

%% Figures

figure('Name',currentDataset)

% Total L1, L2, correlation
subplot(341)
hold on
plot(epochsList, l1normsBBSynth, 'LineWidth', 3)
plot(epochsList, l1normAB, 'LineWidth', 3)
title('Total L1 norms')
% legend('Generated','Defaced')

subplot(342)
hold on
plot(epochsList, l2normsBBSynth, 'LineWidth', 3)
plot(epochsList, l2normAB, 'LineWidth', 3)
title('Total L2 norms')
% legend('Generated','Defaced')

subplot(343)
hold on
plot(epochsList, corrBBSynth, 'LineWidth', 3)
plot(epochsList, corrAB, 'LineWidth', 3)
title('Total correlation')
% legend('Generated', 'Defaced', 'Location', 'northwest')


% Percentiles of L1, L2, correlation
percentileMode = 1;
switch percentileMode
    case 1
        lowPerc = 5;
        highPerc = 95;
    case 2
        lowPerc = 25;
        highPerc = 75;
    case 3
        lowPerc = 33;
        highPerc = 66;
end

% 5% and 95% percentiles for each epoch

meanVal = [median(l1normsBBSynthAll,2), median(l1normABAll,2)];
overMeanVal = cat(3, prctile(l1normsBBSynthAll',highPerc)' - meanVal(:,1), prctile(l1normABAll',highPerc)' - meanVal(:,2));
underMeanVal = cat(3, meanVal(:,1) - prctile(l1normsBBSynthAll',lowPerc)', meanVal(:,2) - prctile(l1normABAll',lowPerc)');
% meanVal = median(l1normsBBSynthAll,2);
% overMeanVal = prctile(l1normsBBSynthAll',highPerc)' - meanVal(:,1);
% underMeanVal = meanVal(:,1) - prctile(l1normsBBSynthAll',lowPerc)';

subplot(345)
hold on
boundedline(epochsList, meanVal, [underMeanVal, overMeanVal],'alpha')
plot(epochsList, meanVal, 'LineWidth', 3)
title('Median L1 norms')
% legend('Generated','Defaced')


meanVal = [median(l2normsBBSynthAll,2), median(l2normABAll,2)];
overMeanVal = cat(3, prctile(l2normsBBSynthAll',highPerc)' - meanVal(:,1), prctile(l2normABAll',highPerc)' - meanVal(:,2));
underMeanVal = cat(3, meanVal(:,1) - prctile(l2normsBBSynthAll',lowPerc)', meanVal(:,2) - prctile(l2normABAll',lowPerc)');
% meanVal = median(l2normsBBSynthAll,2);
% overMeanVal = prctile(l2normsBBSynthAll',highPerc)' - meanVal(:,1);
% underMeanVal = meanVal(:,1) - prctile(l2normsBBSynthAll',lowPerc)';

subplot(346)
hold on
boundedline(epochsList, meanVal, [underMeanVal, overMeanVal],'alpha')
plot(epochsList, meanVal, 'LineWidth', 3)
title('Median L2 norms')
% legend('Generated','Defaced')


meanVal = [median(corrBBSynthAll,2), median(corrABAll,2)];
overMeanVal = cat(3, prctile(corrBBSynthAll',highPerc)' - meanVal(:,1), prctile(corrABAll',highPerc)' - meanVal(:,2));
underMeanVal = cat(3, meanVal(:,1) - prctile(corrBBSynthAll',lowPerc)', meanVal(:,2) - prctile(corrABAll',lowPerc)');
% meanVal = median(corrBBSynthAll,2);
% overMeanVal = prctile(corrBBSynthAll',highPerc)' - meanVal(:,1);
% underMeanVal = meanVal(:,1) - prctile(corrBBSynthAll',lowPerc)';

subplot(347)
hold on
boundedline(epochsList, meanVal, [underMeanVal, overMeanVal],'alpha')
plot(epochsList, meanVal, 'LineWidth', 3)
title('Median correlation')
% legend('Generated', 'Defaced', 'Location', 'northwest')


meanVal = [median(ssimBBSynthAll,2), median(ssimABAll,2)];
overMeanVal = cat(3, prctile(ssimBBSynthAll',highPerc)' - meanVal(:,1), prctile(ssimABAll',highPerc)' - meanVal(:,2));
underMeanVal = cat(3, meanVal(:,1) - prctile(ssimBBSynthAll',lowPerc)', meanVal(:,2) - prctile(ssimABAll',lowPerc)');
% meanVal = median(ssimBBSynthAll,2);
% overMeanVal = prctile(ssimBBSynthAll',highPerc)' - meanVal(:,1);
% underMeanVal = meanVal(:,1) - prctile(ssimBBSynthAll',lowPerc)';

subplot(348)
hold on
boundedline(epochsList, meanVal, [underMeanVal, overMeanVal],'alpha')
plot(epochsList, meanVal, 'LineWidth', 3)
title('Median SSIM')
% legend('Generated', 'Defaced', 'Location', 'northwest')


% Third row: histograms of epoch with best statistic per image
binEdges = [epochsList - 10, epochsList(end)+10];

[~,I1] = min(l1normsBBSynthAll);

subplot(349)
histogram(epochsList(I1),binEdges);
title('Epoch of minimum L1 norm per image')
legend('Generated', 'Location', 'northwest')

[~,I1] = min(l2normsBBSynthAll);

subplot(3,4,10)
histogram(epochsList(I1),binEdges);
title('Epoch of minimum L2 norm per image')
legend('Generated', 'Location', 'northwest')

[~,I1] = max(corrBBSynthAll);

subplot(3,4,11)
histogram(epochsList(I1),binEdges);
title('Epoch of maximum correlation per image')
legend('Generated', 'Location', 'northwest')

[~,I1] = max(ssimBBSynthAll);

subplot(3,4,12)
histogram(epochsList(I1),binEdges);
title('Epoch of maximum SSIM per image')
legend('Generated', 'Location', 'northwest')


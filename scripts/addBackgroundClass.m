

filesA = [dir('testA/*.png'); dir('trainA/*.png')];
filesB = [dir('testB/*.png'); dir('trainB/*.png')];

for i = 1:length(filesA)
    imA = imread(fullfile(filesA(i).folder, filesA(i).name));
    imB = imread(fullfile(filesB(i).folder, filesB(i).name));
   
    thresh = 30;
    imA(imA == 0 & imB > 30) = 127;
    
    fout = fullfile(filesA(i).folder, filesA(i).name);
    
    imwrite(imA, fout);
end
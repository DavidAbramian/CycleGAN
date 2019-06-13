

files = cat(1, dir('*.jpg'), dir('*.gif'), dir('*.tif'), dir('*.tiff'));

for i = 1:length(files)
    im = imread(fullfile(files(i).folder, files(i).name));
    
    [~, fname, ~] = fileparts(files(i).name);
    fout = fullfile(files(i).folder, [fname, '.png']);
    
    imwrite(im, fout);
end
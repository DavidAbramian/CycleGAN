

files = dir('**/*.png');

for i = 1:length(files)
    im = imread(fullfile(files(i).folder, files(i).name));
    
    sz = size(im);
    sz = floor(sz ./ 4) * 4;
    
    im = im(1:sz(1), 1:sz(2));
    
    fout = fullfile(files(i).folder, files(i).name);
    
    imwrite(im, fout);
end
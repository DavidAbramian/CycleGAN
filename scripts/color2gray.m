

files = dir('**/*.png');

for i = 1:length(files)
    im = imread(fullfile(files(i).folder, files(i).name));
    
    if ndims(im) == 3
        im = rgb2gray(im);
    else
        continue
    end
    
    fout = fullfile(files(i).folder, files(i).name);
    
    imwrite(im, fout);
end
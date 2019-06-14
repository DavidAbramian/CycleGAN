

files = dir('**/*.png');

for i = 1:length(files)
    im = imread(fullfile(files(i).folder, files(i).name));
    
    % Check wether image is a label
    if files(i).folder(end) == 'A'
        isLabel = true;
    else
        isLabel = false;
    end
        
    im = imresize(im, 1/2);
    
    % Threshold image if it is a label
    if isLabel
        I = im >= 100;
        im(I) = 255;
        im(~I) = 0;
    end
    
    sz = size(im);
    sz = floor(sz ./ 4) * 4;
    
    im = im(1:sz(1), 1:sz(2));
    
    fout = fullfile(files(i).folder, files(i).name);
    
    imwrite(im, fout);
end
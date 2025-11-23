% Load the template
template_file = 'new.jpg';  
plate_template = imread(template_file);
template_filee = 'OIP.jpg';
plate_templatee = imread(template_filee);
[rows1,cols1] = size(plate_template);
[rows2,cols2] = size(plate_templatee);
kian_constant = (cols2/cols1)*0.66;
disp(kian_constant)

[file, path] = uigetfile({'*.jpg; *.bmp; *.png', 'Choose the image'});
picture = imread(fullfile(path, file));

% Convert images to grayscale (if they are RGB)
if size(picture, 3) == 3
    gray_image = rgb2gray(picture);
else
    gray_image = picture;
end
gray_image = histeq(gray_image);
gray_image = imgaussfilt(gray_image, 2);  % Adjust sigma as needed
gray_image = medfilt2(gray_image);
figure;
if size(plate_template, 3) == 3
    gray_template = rgb2gray(plate_template);
else
    gray_template = plate_template;
end
imshow(gray_image);
% Define the range of scaling factors)
scales = 0.5:0.05:2;
best_corr = -inf;   % initialize best correlation score
best_scale = 1;
best_ypk = 0;
best_xpk = 0;
best_template = gray_template; % best matched (resized) template
% Loop over the scales and compute correlation at each scale
for s = scales
    % Resize the template keeping the aspect ratio constant
    resized_template = imresize(gray_template, s);
    % Compute normalized cross-correlation
    corr = normxcorr2(resized_template, gray_image);
    
    % Find the peak correlation value and its location
    [ypeak, xpeak] = find(corr == max(corr(:)));
    current_corr = max(corr(:));
    
    % Update the best match if current correlation is higher
    if current_corr > best_corr
        best_corr = current_corr;
        best_scale = s;
        best_ypk = ypeak;
        best_xpk = xpeak;
        best_template = resized_template;
    end
end
% Note: normxcorr2 returns coordinates with an offset equal to the template size.
plate_x = best_xpk - size(best_template, 2) + 1;
plate_y = best_ypk - size(best_template, 1) + 1;
plate_width = size(best_template, 2);
plate_height = size(best_template, 1);

figure;
imshow(picture);
hold on;
rectangle('Position', [plate_x, plate_y, plate_width*kian_constant, plate_height], 'EdgeColor', 'r', 'LineWidth', 2);
title('Detected Plate Region');

% Crop the plate region from the original image
plate_region = imcrop(picture, [plate_x, plate_y, plate_width*kian_constant, plate_height]);

% Display the cropped plate region
figure;
imshow(plate_region);
title('Cropped Plate Region');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% from now on it's the
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% same as earlier parts

picture = plate_region
%% 2) now using imresize, we set new dimmensions for the pic we picked
NUMROWS = 300;
NUMCOLS = 500;
picture = imresize(picture,[NUMROWS,NUMCOLS]);
%% 3) now a function to play with its third dimmension channels (RGB) to make it grayscale
function gray_picture = mygrayfun (picture)
    NUMROWS = 300;
    NUMCOLS = 500;
    gray_picture = zeros(NUMROWS, NUMCOLS, 'uint8'); % I created a two dimmensional matrix to store those gray values 
    for i = 1:1:NUMROWS 
        for j = 1:1:NUMCOLS
            gray_picture(i,j) = 0.299*picture(i,j,1) + 0.578*picture(i,j,2) + 0.114*picture(i,j,3);
        end
    end
end
new_picture = mygrayfun(picture);
figure;
imshow(mygrayfun(picture));
%% 4) now defining a func to make it from grayscale to binary by setting a threshold
function out = mybinaryfun(picture, threshold)
    [NUMROWS, NUMCOLS] = size(picture,1,2);
    binary = zeros(NUMROWS,NUMCOLS);
    for i = 1:1:NUMROWS
        for j = 1:1:NUMCOLS
            if picture(i,j) > threshold
                binary(i,j) = 1;
            else
                binary(i,j) = 0;
            end
        end
    end
    out = binary;
end
new_binary = mybinaryfun(new_picture, 100);
figure;
imshow(new_binary);
%% extra) part I added : change each ones with zeros and vice versa :
function out = swap_zeros_and_ones(input_matrix)
    if islogical(input_matrix)
        out = ~input_matrix;
    else
        out = input_matrix;
        out(input_matrix == 0) = 1;
        out(input_matrix == 1) = 0;
    end
end
new_binary = swap_zeros_and_ones(new_binary);
figure;
imshow(new_binary);
%% 5) I am asked to implement bwareaopen function myslef 
function cleaned_image = removemycom(binary_image, n)
    [rows, cols] = size(binary_image);
    cleaned_image = binary_image;
    visited = false(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            if cleaned_image(i, j) == 1 && ~visited(i, j)
                queue = [i, j];
                object_pixels = [];
                visited(i, j) = true;
                neighbors = [-1, 0; 1, 0; 0, -1; 0, 1];
                
                while ~isempty(queue)
                    current = queue(1, :);
                    queue(1, :) = [];
                    object_pixels = [object_pixels; current];
                    
                    for k = 1:size(neighbors, 1)
                        ni = current(1) + neighbors(k, 1);
                        nj = current(2) + neighbors(k, 2);
                        
                        if ni >= 1 && ni <= rows && nj >= 1 && nj <= cols ...
                                && cleaned_image(ni, nj) == 1 && ~visited(ni, nj)
                            visited(ni, nj) = true;
                            queue = [queue; ni, nj];
                        end
                    end
                end
                
                if numel(object_pixels) < n
                    for p = 1:size(object_pixels, 1)
                        cleaned_image(object_pixels(p, 1), object_pixels(p, 2)) = 0;
                    end
                end
            end
        end
    end
end
figure;
new_cleaned = removemycom(new_binary,1200);
imshow(new_cleaned);
%% now implementing bwlabel, it will get the picture, and then output first
% the new labled imgae where each connected ones have their own unique intege
% value as label, and the second output would be the number of labled
% groups
function [L, Ne] = mysegmentation(picture)
    [rows, cols] = size(picture);
    
    % Initialize labeling matrix
    labeling_matrix = zeros(rows, cols);
    label = 1; % Start labels from 1
    
    % To handle label equivalences
    equivalence = containers.Map('KeyType', 'double', 'ValueType', 'double');
    
    % First pass: Assign initial labels and record equivalences
    for i = 1:rows
        for j = 1:cols
            if picture(i, j) == 1
                % Collect labels of connected neighbors
                neighbors = [];
                
                % Check top neighbor
                if i > 1 && picture(i-1, j) == 1
                    neighbors = [neighbors, labeling_matrix(i-1, j)];
                end
                
                % Check left neighbor
                if j > 1 && picture(i, j-1) == 1
                    neighbors = [neighbors, labeling_matrix(i, j-1)];
                end
                
                % Assign the smallest label or a new label
                if isempty(neighbors)
                    labeling_matrix(i, j) = label;
                    label = label + 1;
                else
                    min_label = min(neighbors);
                    labeling_matrix(i, j) = min_label;
                    
                    % Record equivalences
                    for neighbor_label = neighbors
                        if neighbor_label ~= min_label
                            equivalence(neighbor_label) = min_label;
                        end
                    end
                end
            end
        end
    end
    
    % Resolve label equivalences (Union-Find)
    function root = find_root(l)
        while isKey(equivalence, l)
            l = equivalence(l);
        end
        root = l;
    end
    
    % Second pass: Resolve equivalences and assign final labels
    for i = 1:rows
        for j = 1:cols
            if labeling_matrix(i, j) > 0
                labeling_matrix(i, j) = find_root(labeling_matrix(i, j));
            end
        end
    end
    
    % Return the labeled image and the number of connected components
    L = labeling_matrix;
    Ne = max(L(:)); % Number of connected components
end





% ========================================================
% STEP 1: LOAD REFERENCE IMAGES FROM THE Map_Set DIRECTORY
% ========================================================
% ========================================================
% STEP 1: LOAD REFERENCE IMAGES FROM THE Map_Set DIRECTORY
% ========================================================
map_set_dir = 'persian_mapset'; % Directory containing reference images (letters/numbers)

% Get a list of all .png and .bmp files in the directory
file_list_png = dir(fullfile(map_set_dir, '*.png')); % Get .png files
file_list_bmp = dir(fullfile(map_set_dir, '*.bmp')); % Get .bmp files

% Combine the file lists
file_list = [file_list_png; file_list_bmp];

% Initialize the TRAIN cell array: Row 1 = Images, Row 2 = Labels
TRAIN = cell(2, numel(file_list));

for k = 1:numel(file_list)
    % Load image and preprocess
    img = imread(fullfile(map_set_dir, file_list(k).name));
    if size(img, 3) == 3
        img = im2bw(img); % Convert RGB to binary
    end
    img = imresize(img, [42, 24]); % Resize to match character size
    
    % Store in TRAIN
    TRAIN{1, k} = img; % Image
    [~, label, ~] = fileparts(file_list(k).name); % Extract label (filename without extension)
    TRAIN{2, k} = label;
end

% Display the number of images loaded
fprintf('Loaded %d images from the Map_Set directory.\n', numel(file_list));

% ========================================================
% STEP 2: LICENSE PLATE SEGMENTATION AND RECOGNITION
% ========================================================
[L, Ne] = mysegmentation(new_cleaned);
propied = regionprops(L, 'BoundingBox', 'Area');

% Sort bounding boxes by x-coordinate (left-to-right)
boundingBoxes = cat(1, propied.BoundingBox);
[~, sortOrder] = sort(boundingBoxes(:, 1));
sorted_propied = propied(sortOrder);

% Filter out unwanted regions (e.g., outer border)
valid_indices = [];
for n = 1:Ne
    bbox = sorted_propied(n).BoundingBox;
    width = bbox(3);
    height = bbox(4);
    area = width * height;
    
    % Criteria (adjust these thresholds)
    max_area_threshold = 15000;
    min_area_threshold = 1450;
    aspect_ratio_range = [0.1, 4]; % [min_aspect_ratio, max_aspect_ratio]
    
    % Calculate aspect ratio
    aspect_ratio = width / height;
    
    % Check if the region meets all criteria
    if area > min_area_threshold && area < max_area_threshold && ...
       aspect_ratio >= aspect_ratio_range(1) && aspect_ratio <= aspect_ratio_range(2)
        valid_indices = [valid_indices, n]; % Add index to valid regions
    end
end

filtered_propied = sorted_propied(valid_indices);
filtered_Ne = numel(valid_indices);

% Draw bounding boxes
figure;
imshow(new_cleaned);
hold on;
for n = 1:filtered_Ne
    rectangle('Position', filtered_propied(n).BoundingBox, 'EdgeColor', [0.7,0,0.7], 'LineWidth', 2);
end
hold off;

% Recognize characters
final_output = [];
for n = 1:filtered_Ne
    region_idx = valid_indices(n);
    [r, c] = find(L == sortOrder(region_idx)); 
    
    if isempty(r) || isempty(c)
        continue;
    end
    
    Y = new_cleaned(min(r):max(r), min(c):max(c));
    
    if isempty(Y)
        continue;
    end
    
    Y = imresize(Y, [42, 24]);
    
    ro = zeros(1, size(TRAIN, 2));
    for k = 1:size(TRAIN, 2)
        ro(k) = corr2(TRAIN{1, k}, Y);
    end
    
    [MAXRO, pos] = max(ro);
    if MAXRO > 0.45
        out = cell2mat(TRAIN(2, pos));
        final_output = [final_output out];
    end
end

disp('Recognized Output:');
disp(final_output);

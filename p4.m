template_file = 'new.jpg';  
plate_template = imread(template_file);

[file, path] = uigetfile({'*.mp4;*.avi'}, 'Select Video File');
video_path = fullfile(path, file);
video = VideoReader(video_path);

known_distance = 0.40; % 40 cm in meters
t1 = 0.0;  
t2 = 1.5;  

frame1 = read(video, round(t1 * video.FrameRate) + 1);
frame2 = read(video, round(t2 * video.FrameRate) + 1);

[scale1, proc_frame1, rect1] = process_frame(frame1, plate_template);
[scale2, proc_frame2, rect2] = process_frame(frame2, plate_template);

% Calibrate constant using known distance and measured scales
kian_constant = known_distance / (1/scale2 - 1/scale1);

time_interval = t2 - t1;
speed = calculate_speed(scale1, scale2, kian_constant, time_interval);

disp(['Estimated Speed: ', num2str(speed), ' km/h']);

figure;
subplot(2,2,1);
imshow(frame1);
if ~isempty(rect1)
    rectangle('Position', rect1, 'EdgeColor', 'r', 'LineWidth', 2);
end
title('Frame 1');

subplot(2,2,2);
imshow(frame2);
if ~isempty(rect2)
    rectangle('Position', rect2, 'EdgeColor', 'r', 'LineWidth', 2);
end
title('Frame 2');

if ~isempty(rect1)
    plate_region1 = imcrop(frame1, rect1);
    subplot(2,2,3);
    imshow(plate_region1);
    title('Plate Region from Frame 1');
end
if ~isempty(rect2)
    plate_region2 = imcrop(frame2, rect2);
    subplot(2,2,4);
    imshow(plate_region2);
    title('Plate Region from Frame 2');
end

function [best_scale, gray_image, best_match_rect] = process_frame(picture, plate_template)
    if size(picture, 3) == 3
        gray_image = rgb2gray(picture);
    else
        gray_image = picture;
    end
    gray_image = histeq(gray_image);
    gray_image = imgaussfilt(gray_image, 2);
    gray_image = medfilt2(gray_image);

    if size(plate_template, 3) == 3
        gray_template = rgb2gray(plate_template);
    else
        gray_template = plate_template;
    end

    scales = 0.5:0.05:2;
    best_corr = -inf;
    best_scale = 1;
    best_match_rect = [];
    
    for s = scales
        resized_template = imresize(gray_template, s);
        corr = normxcorr2(resized_template, gray_image);
        current_corr = max(corr(:));
        
        if current_corr > best_corr
            best_corr = current_corr;
            best_scale = s;
            [ypeak, xpeak] = find(corr == current_corr, 1);
            yoff = ypeak - size(resized_template,1);
            xoff = xpeak - size(resized_template,2);
            best_match_rect = [xoff+1, yoff+1, size(resized_template,2), size(resized_template,1)];
        end
    end
end

function speed = calculate_speed(scale1, scale2, k_const, dt)
    distance1 = k_const / scale1;
    distance2 = k_const / scale2;
    speed = abs(distance2 - distance1) / dt * 3.6; % Convert m/s to km/h
end

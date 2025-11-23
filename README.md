License Plate Recognition & Vehicle Speed Estimation

Signal & Systems
Author: Kian Fotovat · SID: 810102486

Overview

This project implements a complete image-processing pipeline for recognizing characters on vehicle license plates and estimating the speed of a moving vehicle using frame-to-frame scale changes.
The work is developed entirely in MATLAB, without relying on built-in labeling or segmentation functions such as bwlabel or bwareaopen. Core operations—grayscale conversion, binarization, noise removal, segmentation, character matching, and plate extraction—are implemented manually.

The project is divided into four parts:

Preprocessing, noise removal, segmentation, and English plate recognition

Persian plate recognition

Template-based plate extraction from a car image

Speed estimation using multi-frame analysis

Features
✔ Custom Image Preprocessing

Manual RGB → Grayscale conversion (weighted sum)

Adjustable resizing parameters

Custom binary conversion with tunable thresholds

Optional image enhancement using histogram equalization, Gaussian filtering, and median filtering

Noise removal using a BFS-based connected-component detector (manual bwareaopen)

✔ Custom Connected-Component Labeling

Full two-pass segmentation implemented manually

Label equivalence resolved using a dictionary-based union-find

Accurate bounding-box extraction without MATLAB’s bwlabel

✔ Character Recognition

Loads reference characters from Map_Set/ or persian_matset/

Resizes segmented regions to a uniform size

Uses corr2 to compute similarity with reference characters

Applies correlation thresholds to avoid false positives

Outputs the final plate string into a .txt file

✔ Plate Extraction with Template Matching

Uses a known blue plate-edge template

Performs multi-scale template matching via normxcorr2

Tracks the best scale and correlation peak

Computes a custom scale multiplier (kian_constant) to expand the bounding box from the flag region to the entire plate

Crops and visualizes the detected plate region

✔ Vehicle Speed Estimation

Reads two frames from a video (.mp4/.avi)

Detects the plate in each frame using the same template-matching pipeline

Estimates relative distance using scale differences

Computes vehicle speed from temporal changes

Displays detected plate regions and the final speed in km/h

File Structure
/project
│
├── p1.m                 % Main script for Part 1
├── p2.m                 % Persian plate recognition (Part 2)
├── p3.m                 % Template-based plate extraction
├── p4.m                 % Speed estimation module
│
├── Map_Set/            % English reference characters
├── persian_matset/     % Persian reference characters (F-English)
│
├── new.jpg             % Template for plate edge
├── OIP.jpg             % Example plate reference
│
└── output.txt          % Recognized license plate (generated)

How It Works – High-Level Pipeline
1. Image Preprocessing

Load input image

Resize

Convert to grayscale

Convert to binary with thresholding

Optionally invert binary image

Apply noise removal via BFS

Result: a clean binary image ready for segmentation

2. Segmentation (Manual bwlabel)

Scan pixels row-by-row

Assign labels based on top/left neighbors

Track label equivalences

Second pass: unify labels through union-find

Compute bounding boxes via regionprops

3. Selecting Valid Character Regions

Filters applied:

Minimum/maximum area threshold

Aspect ratio range (0.1 → 4)

Sorted left-to-right by x-coordinate

Valid segments are boxed and prepared for recognition.

4. Character Recognition

Resize each segment to 42×24

Compare with every reference image using corr2

Choose the highest-correlation result

Apply a threshold (>0.45)

Append the recognized character to the output

5. Plate Extraction (Part 3)

Load template of plate’s left-side emblem

Rescale the template across a range (0.5 → 2 in steps of 0.05)

Use normalized cross-correlation to track the best match

Expand bounding box horizontally using kian_constant

Crop the full plate image

6. Speed Estimation (Part 4)

Extract two frames at t1 and t2

Detect plate scale in each frame

Relate scale change to distance

Use a known real-world measurement (40 cm) for calibration

Compute speed using:

distance = k_const / scale
speed = |d2 - d1| / Δt  * 3.6   % km/h

Dependencies

MATLAB R2020+

Image Processing Toolbox (used only for basic ops like imshow, imresize, normxcorr2, regionprops, etc.)

No Python or external libraries required.

Usage
Run English Plate Recognition
run('p1.m')

Run Persian Plate Recognition
run('p2.m')

Extract Plate From Image
run('p3.m')

Estimate Speed From Video
run('p4.m')


Output is written to output.txt or displayed in MATLAB’s command window.

Notes

Persian characters are represented in F-English (e.g., “م → M”, “ب → B”).

Threshold values for segmentation and matching may need tuning for different lighting conditions.

Template-matching performance depends on the similarity of the plate to the provided template.

License

This project is provided for academic and educational use
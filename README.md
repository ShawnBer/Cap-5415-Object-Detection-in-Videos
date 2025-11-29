# Cap-5415-Object-Detection-in-Videos

# Necessary Packages:
ultralytics, numpy, cv2, os, argparse, glob, sys, typing, yaml, pytorch

# Deploying the yolo11 model for object detection within videos

VideoObjectDetection.py and VideoObjectDetection_Tracking.py how to run:

In both of these .py files they have the same initialization steps in order to get them working.
Steps:
1. First thing is make sure that you have the correct file pathing of one of our training runs from yolo set in **line 8 for VideoObjectDetection.py** and in **line 88 for VideObjectDetection_Tracking.py** 
(for example: 'runs/detect/Sixth_Train/weights/best.pt')
2. Second thing is to put a video you want to test from virat within the folder pathing, **Videos/Original/**. And then in **line 11 for VideoObjectDetection.py** and in **line 92 for VideObjectDetection_Tracking.py** update the line to reflect the name of the video you uploaded
(for example: 'Videos/Original/VIRAT_S_010001_09_000921_000952.mp4')
3. Third thing is that you should change the last part of the output path in **line 14 for VideoObjectDetection.py** and in **line 95 for VideObjectDetection_Tracking.py** between different videos if you don't want the outupts to overwrite eachother, 'Videos/Model_Output/some_name.mp4'
4. Once those three things have been checked you can simply run either .py file and you should have a window pop up that displays the inferences as they are happening
5. Finally, you can either press 'q' or wait for the video to fully finish to see a saved version of the inferences video in 'Videos/Model_Output/whatever_you_named_it.mp4'

# Within the folder Creating Frames and Annotations we have three .py files that we utilized for getting frames and annotations from VIRAT

getting_annotations.py, getting_frames.py, visualize_yolo_labels.py how to run:

Within this folder there is 3 .py files, getting_annotations.py, getting_frames.py, visualize_yolo_labels.py

Utilizing getting_annotations.py:
Steps:
1. The first thing to do is put the annotation/annotations from VIRAT you want to get the annotations of in 'Creating Frames and Annotations/original_annotations'
2. Then open up the .py file and edit the **width and height lines 17 and 18** to reflect the width and height of the input annotations original video dimensions
3. Then from there you should be able to run the .py file and it should start printing the number of annotations it is on and will put the yolo format annotations in the folder 'Creating Frames and Annotations/annotations_for_frames'

Utilizing getting_frames.py:
Steps:
1. The first thing to do is put the video/videos from VIRAT you want to get the frames of in 'Creating Frames and Annotations/input_videos'
2. Then from there you should be able to run the .py file and it should start printing the .jpg frame it has created and will put the framecount.jpg images in the folder 'Creating Frames and Annotations/frames_output'

These two above will create matching set of frames with annotations in YOLO format from the VIRAT dataset.

# Training YOLO 11

Utilizing yolo11_training.py:
Steps:

1. You can try to run it instantly by just running the .py file
2. It will automaticaly download the YOLO files needed and the model that is specified which is, yolo_11s
3. We decided to utilize custom data augmentation command for our training that you can see and edit if needed

4. It will most likely crash unless you have around 16gb of VRAM, so if you want to run a less intense training you can change the imgsz variable to 640 as it being 960 greatly increases its computational cost

# Original Dataset that we modified
https://github.com/hayatkhan8660-maker/OD-VIRAT
This is the citation they have but I don't know if this is right: MMDetection Contributors. (2018). OpenMMLab Detection Toolbox and Benchmark [Computer software]. https://github.com/open-mmlab/mmdetection

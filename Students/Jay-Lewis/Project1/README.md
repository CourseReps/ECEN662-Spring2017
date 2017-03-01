# ECEN662_group_project1

Creator: TDW

Date: Feb 27 2017

Shared files for group project 1.
May change as code is updated.

## Contents
* edgeDetectionSummary.ipnyb - summary of work, includes the python contents of edgeMain.py
* img1-3.png - image files used in summary of work
* edgeMain.py - minimum python file for generating the results in the summary
* loadImg.py - python file that handles image loading
* settings.ini - specifies the relative filepath to the image root directory

## How to use the code
* Download the training data sets and put them in a common folder; e.g., '~/ecen/TrainingSetSynthetic' and '~/ecen/TrainingSetScenes'
* Modify settings.ini so that the filepath variable is the relative filepath from the code directory to the root directory for the images, e.g. '~/ecen'
* Run 'python loadImg' to run the test code and load two images
* Run 'python edgeMain' to generate the plots and analysis used in the summary of work

Code tested in Python 2.7 and 3.5.
For 2.7 Spyder packages were used, for 3.5 Anaconda packages.

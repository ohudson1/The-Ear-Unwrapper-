## Ear Unwrapper Pipeline
Owen Hudson, Colin Brahmstedt, Dylan Hudson, and Jeremy Brawner

If you find this resource useful, please cite:

The Ear Unwrapper is a machine that was designed to image ears of corn/maize. If adjusted and calibrated appropriately for differences in size, other cylindrical objects can also be imaged with the line-scan unwrapping technique.

## Design and Function:
The Ear Unwrapper performs a line-scan to image a full cylindrical object into a 'flat' or 'unwrapped' version allowing the entire surface of the object to be analyzed. A stepper motor sequentially rotates an ear while taking a series of images at highly accurate intervals, dewarping the images, then slicing a 'line' of pixels from each, and combining these thin lines into a single composite image. The images are then scaled to reflect their relative size, as natural changes in corn ear diameter cause stretching or compression when using a constant number of steps/pixels, then processed with a random forest model via the [Ilastik](https://www.ilastik.org/) biological image processing suite for semantic pixel classification. Post-processing of the probabilistic semantic pixel maps includes cropping, blur, then thresholding for mask creation for subsequent quantitative analysis. 

## Usage of Image Acquisition and Post-Processing Scripts

Requirements:
Python version > 3.6 (possibly compatible with previous versions, but not fully tested below 3.6)

Image acquisition and camera interfacing has only been tested on Ubuntu-based Linux distros, and requires the installation of OpenUSB:
sudo apt install libopenusb-dev
 
Required python library installation with pip:
pip install opencv-python numpy pyusb pandas

The image acquisition application is invoked with one command-line argument -–idnum, a unique integer identifier that we used to sequentially identify individual samples-
python3 Ear_unwrapper_acquisition_script.py --idnum 42

The post-processing scripts assume the formatting of filename metadata is consistent with the output filenames specified in the acquisition script, and the csv header names as specified in the code. These can be adapted if necessary. 

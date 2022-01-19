# Knee segmentation
This is repository contains my CNN that covers topic of segmenting knee MR images into
6 sections: femur bone, tibial bone, kneecap, posterior cruciate ligaments, anterior cruciate ligaments, background.
It contains original images with manualy segmented sections and scripts to work on this data.


# Folders structure
Original images are in "png" folder, they were taken out of DICOM images and saved as .png.
Contours made manualy are in "contours" folder, they have the same names as the coresponding images in "png".
Contours that are prepared to be used in CNN are contained in "prepared" folder.
With using data augumentation you can generate images and contours that will be stored in "png2" and "prepared2" folders.
While teaching CNN, it's models and weights will be stored in "models" folder. This folder is split into 4 subfolders that coresponds to different method of data augmentation used. Then each of those folders contains folders 1-3 coresponding to deepness of CNN used. Inside of each of those folders are stored weights of models as well as folders for results on validation sets.
Folder "result" contains result of TestCNN script used on model stored in "model".

# Workflow
First use script called PrepareContours.py to change images from "png" from (0, 255) range into (0, 5) range meaning classes of segmentation.
Then use DataAugumentation.py script and by specifying howMany variable choose how many images should be generated (howMany=5 means 5x original amount of images).
After that all images to teach are now prepared and we are ready to use TeachCNN.py. At the start of the file there are specified parameters that  can be changed to alter the model. After running this file will generate model, it's weights in different stages and plot showing how it was learning. It uses k-fold cross validation so it will run whole learning more then once. After it finishes it removes unwanted saved weights. 
Now if you want to view models results on validation images just go to coresponding folder => augumentation_type//deepness//n_fold
If you are intrested in statistical data run CreateResults.py file that will accumulate data from results and compare it to true images, calculating IoU metric.

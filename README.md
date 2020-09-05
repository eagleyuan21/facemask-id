# facemask-id

Modern problems require modern solutions, so during the COVID-19 pandemic, one useful tool to fight back against the virus is wearing a face covering. This project focuses on using python to create a tool to help quickly identify correct mask wearing, wrong mask wearing, or no mask wearing.

Currently, the project can only identify mask covering or no mask covering. train.py is the file that trains the Convolutional Neural Network (CNN), which currently determines only mask or no mask wearing, and main.py utilizes the model and the .xml file to identify subjects in the camera feed and provide live time feedback. The model present includes a 100 pixel version. However, it would be recomended to use a 200 px version by changing img_size in train.py to 200. The 200 px version is more accurate as it uses more data points during resizing, whereas the 100 px version would leave out more data. However, the 200 px run time in both train.py and main.py takes significantly longer due to more data, so it is more optimal to use 100 px. To be able to run, make sure all the import python packages are downloaded.

The dataset folder has three folders with around 1800 images each of right and no mask wearing and around 100 images of wrong mask wearing. The wrong mask wearing folder contains much less pictures because it is hard to find wrong mask wearing images. Instead, the ps folder contains a set of around 1800 images of no mask wearing along with a folder of mask .png images. The task is to photoshop wrong mask wearing onto these no mask wearing pictures and then add these photoshopped images into the dataset/wrong folder. Once these are completed, the train.py file will be updated to produce 3 output nodes, and the main.py file will also be updated to be able to recognize the 3 output nodes.

This project is based of the project found here at https://github.com/aieml/face-mask-detection-keras. The main differences are that the dataset here is compiled and edited by myself, the CNN is trained in color instead of black and white (uses three values representing RGB saturation), the CNN is able to recognize most types of face coverings instead of just a disposable surgical mask, and the eventual addition of the ability to identify wrong mask wearing, specifically failure to cover the nose. 

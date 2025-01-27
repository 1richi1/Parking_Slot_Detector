
# Parking_Slot_Detector
Parking Slot Detection Approaches

----------------------------------------------------------
| FINAL PROJECT -- PARKING SLOTS DETECTOR AND CLASSIFIER |
----------------------------------------------------------

General structure:

1)  Data folder contains all the testing images.
    Add in this folder the images you want to test.

2)  Results folder contains the results from the developed approaches. 
    You can find each single result, the merged version and the final optimized one.
    The name of each folder corresponds to the name of the image you have added in the data folder.

3)  CNR-EXT, CNRPark folders are the datasets used for the trainig part --> see the config.png file to configure correctly
    Substantially, you need to put these two folders in the root folder. This is needed to make the .csv file working
    correctly. So is something is not clear, open the .csv file and follow the path of the images to recreate the structure.

4)  mlp_model.pkl file is the pre-trained MLP model. You can save time  by using it and testing directly
    the algorithms' performances.

To run the project open the main.py file and run it. It is already configured in test mode, so it takes
the mlp_model.pkl file and compute the results on the given image. When you have added the image in the data folder,
in the main.py file change the filepath variable with the path of your image.
If you want to train or tune the MLP model, uncomment the corresponding line in the main.py file and change 
the configuration in the neural_network.py file.

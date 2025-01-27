from functions import *
from imports import *

def selective_search_test(mlp: MLPClassifier() , filepath: str):
    '''
        2nd approach:
        Using selective search function to find possible regions
         - Take the contours of the image and apply selective search algorithm
         - Make the list of cropped images and use a pre-trained mlp to predict each one
    '''
    image = cv2.imread(filepath)
    #image = image.astype(np.uint8)
    _, regions = selectivesearch.selective_search(
        image, scale = 1000 , sigma=0.7, min_size=int(np.ceil(image.shape[0]/10)))

    candidates = first_filtering_layer(regions , area= (image.shape[0]*image.shape[1]))

    # Sort the set based on the area and remove the last one --> it's the box of all the image
    sortset =sorted(candidates, key=lambda x: x[-2] * x[-1])
    #sortset.pop()

    # Filter out boxes that are completely inside others
    filtered_boxes = filter_inside_boxes(sortset)

    # Display first results
    #display_boxes_on_image(image,filtered_boxes)

    croppedImages = cropping_image(image , filtered_boxes)

    siftarray = sift_computation(croppedImages)

    pred = mlp.predict(siftarray)

    # Results of first test with selective search: red occupied green free
    #results_with_pred(image, filtered_boxes , pred)

    return filtered_boxes , pred

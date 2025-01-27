from functions import *
from imports import *

def contours_search(mlp: MLPClassifier() , filepath: str):
    '''
        1st TEST: minAreaRect algorithm
        - find contours
        - apply min area algorithm
        - filter redundant boxes
    '''
    image = cv2.imread(filepath)
    #sift = cv2.SIFT_create()

    # Find contours
    contours = find_contours(image)
    # Filter it
    filtered_boxcontainer = filter_contours_min_area(contours=contours , area=image.shape[0]*image.shape[1])
    # Sift extraction
    siftarrayelems = cropping_and_sift_min_area_algorithm(image= image , filtered_boxcontainer= filtered_boxcontainer)
    # Predict
    pred = mlp.predict(siftarrayelems)

    # FILTERING USING A DISTRIBUTION PERCENTILE
    valid_cntlist, valid_indexlist = filter_boxes_by_area(filtered_boxcontainer, pred)

    for cnt, index in zip(valid_cntlist, valid_indexlist):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        if index == '1':
            cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        else:
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    #show_img(img= image , filepath= "None")

    return filtered_boxcontainer , pred , valid_cntlist , valid_indexlist
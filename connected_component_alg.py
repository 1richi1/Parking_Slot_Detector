from functions import *
from imports import *

def connected_component_alg(mlp: MLPClassifier() , filepath: str):
    '''
        3rd solution: scikit-image algorithm
        - connected component labeling algorithm
    '''
    image = cv2.imread(filepath)
    imagenew = color.rgb2gray(image)

    # apply threshold
    thresh = threshold_otsu(imagenew)
    bw = closing(imagenew > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    newregionprops = []
    newImages = []
    for region in regionprops(label_image):
        if region.area >= 10:
            newregionprops.append(region)
            minr, minc, maxr, maxc = region.bbox
            grayversion = cv2.cvtColor(image[minr:maxr, minc:maxc],cv2.COLOR_BGR2GRAY)
            newImages.append(grayversion)

    siftarrayelems3 = sift_computation(newImages)

    pred3 = mlp.predict(siftarrayelems3)

    areas = []
    for region in newregionprops:
        minr, minc, maxr, maxc = region.bbox
        areas.append(np.abs((maxc-minc)*(maxr-minr)))
    
    threshold = np.percentile(areas, 70)

    validnewregionprops = []
    validpred3 = []

    for region, index in zip(newregionprops, pred3):
        minr, minc, maxr, maxc = region.bbox
        area = np.abs((maxc-minc)*(maxr-minr))

        if area > threshold:
            validnewregionprops.append(region)
            validpred3.append(index)


    for region , index in zip(validnewregionprops,validpred3):
        minr, minc, maxr, maxc = region.bbox

        if index == '0':
            cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 0), 2) 
        elif index == '1':
            cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
            
    #show_img(img=image , filepath= "None" , title='th')

    return validnewregionprops , validpred3
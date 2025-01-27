from functions import *
from imports import *

def show_results(filepath : str , filtered_boxes: list , pred: list , filtered_boxcontainer: list , pred2: list , regionprops_label_image:list , pred3:list , valid_cntlist: list , valid_indexlist: list ):
    totalimage = cv2.imread(filepath)
    totalboxes = []
    totalpreds = []

    ##############################################################
    # 1st sol
    first_results_middle = cv2.imread(filepath)
    first_results = cv2.imread(filepath)
    for cnt , index in zip(valid_cntlist,valid_indexlist):
        # Get the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(cnt)
        # Convert the rectangle to a box (4 points)
        box = cv2.boxPoints(rect)
        # Round the coordinates to integers
        box = np.intp(box)
        x, y, w, h = cv2.boundingRect(box)
        totalboxes.append((x,y,w,h))
        totalpreds.append(index)
        # Draw the box on the image
        if index == '0':
            cv2.drawContours(first_results_middle, [box], 0, (0, 255, 0), 2)
            cv2.rectangle(first_results, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(totalimage, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif index == '1':
            cv2.drawContours(first_results_middle, [box], 0, (0, 0, 255), 2)
            cv2.rectangle(first_results, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(totalimage, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_img(img=first_results_middle , filepath= filepath , title='Fisrt_solution_min_area')
    show_img(img=first_results , filepath= filepath , title='First_solution')

    ##############################################################
    # 2nd sol
    seond_results = cv2.imread(filepath)
    for (x, y, w, h) , index in zip(filtered_boxes,pred):
        totalboxes.append((x,y,w,h))
        totalpreds.append(index)
        if index == '0':
            cv2.rectangle(seond_results, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(totalimage, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif index == '1':
            cv2.rectangle(seond_results, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(totalimage, (x, y), (x + w, y + h), (0, 0, 255), 2)

    show_img(img= seond_results , filepath=filepath , title='Second_solution')

    ##############################################################
    #3rd sol
    third_results = cv2.imread(filepath)
    for region , index in zip(regionprops_label_image,pred3):
        minr, minc, maxr, maxc = region.bbox
        x, y, w, h = minc, minr, maxc - minc, maxr - minr
        #if(w*h>=(0.01*totalimage.shape[0]*totalimage.shape[1]) and w*h<=(0.30*totalimage.shape[0]*totalimage.shape[1])):
        totalboxes.append((x,y,w,h))
        totalpreds.append(index)
        if index == '0':
            cv2.rectangle(third_results, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            cv2.rectangle(totalimage, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
        elif index == '1':
            cv2.rectangle(third_results, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
            cv2.rectangle(totalimage, (minc, minr), (maxc, maxr), (0, 0, 255), 2)  
    show_img(img=third_results , filepath= filepath , title= 'Third_solution')
    
    
    ##############################################################
    # Otpimizing the results
    # Before optimization 
    show_img(img=totalimage , filepath= filepath , title= 'All_together') 
    
    # 1st step : REMOVING GREEN BOXES INSIDE OTHER GREEN BOXES

    to_remove = []  # List to store indices of boxes to be removed

    for i, (box1, pred1) in enumerate(zip(totalboxes, totalpreds)):
        if pred1 == '0':  # Check if the box is green
            for j, (box2, pred2) in enumerate(zip(totalboxes, totalpreds)):
                if i != j and pred2 == '0':  # Avoid comparing the box with itself and check if the other box is also green
                    x1, y1, w1, h1 = box1
                    x2, y2, w2, h2 = box2
                    # Check if box1 is fully contained in box2
                    if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                        to_remove.append(i)
    # Remove smaller green boxes
    newboxes = [box for i, box in enumerate(totalboxes) if i not in to_remove]
    newpreds = [pred for i, pred in enumerate(totalpreds) if i not in to_remove]

    # 2nd step: REMOVE RED INTO GREEN
    to_remove_red = []  # List to store indices of red boxes containing green boxes

    for i, (box1, pred1) in enumerate(zip(newboxes, newpreds)):
        red_contains_green = False
        red_contains_red = False
        for j, (box2, pred2) in enumerate(zip(newboxes, newpreds)):
            if i != j:
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                # Check if box1 (red) contains at least one green box
                if pred2 == '0' and x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2:
                    red_contains_green = True
                # Check if box1 (red) contains at least one red box
                elif pred2 == '1' and x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2:
                    red_contains_red = True
        
        # If box1 (red) contains at least one green and one red box, mark it for removal
        if red_contains_green and red_contains_red:
            to_remove_red.append(i)

    # Remove red boxes containing green boxes
    secondnewboxes = [box for i, box in enumerate(newboxes) if i not in to_remove_red]
    secondnewpreds = [pred for i, pred in enumerate(newpreds) if i not in to_remove_red]



    # 3rd step : REMOVE GREEN BOXES THAT OVERLAP MORE THAN 40% OF ITS AREA WITH RED BOXES
    to_remove_green = []  # List to store indices of green boxes with overlap

    for i, (box1, pred1) in enumerate(zip(secondnewboxes, secondnewpreds)):
        if pred1 == '0':  # Check if the box is green
            for j, (box2, pred2) in enumerate(zip(secondnewboxes, secondnewpreds)):
                if i != j and pred2 == '1':  # Check if the other box is red
                    overlap_percentage = calculate_overlap(box1, box2)
                    if overlap_percentage >= 40:
                        to_remove_green.append(i)

    # Remove green boxes with at least 40% overlap with red boxes
    thirdnewboxes = [box for i, box in enumerate(secondnewboxes) if i not in to_remove_green]
    thirdnewpreds = [pred for i, pred in enumerate(secondnewpreds) if i not in to_remove_green]

    # 4th step : REMOVE GREEN THAT OVERLAP MORE THAN 80% ANOTHER GREEN
    to_remove_green = []  # List to store indices of green boxes with overlap

    for i, (box1, pred1) in enumerate(zip(thirdnewboxes, thirdnewpreds)):
        if pred1 == '0':  # Check if the box is green
            for j, (box2, pred2) in enumerate(zip(thirdnewboxes, thirdnewpreds)):
                if i != j and pred2 == '0':  # Check if the other box is also green
                    overlap_percentage = calculate_overlap(box1, box2)
                    if overlap_percentage >= 80:
                        # Remove the smaller green box
                        area1 = box1[2] * box1[3]
                        area2 = box2[2] * box2[3]
                        if area1 < area2:
                            to_remove_green.append(i)
                        else:
                            to_remove_green.append(j)

    # Remove green boxes with at least 80% overlap with another green box
    fourthnewboxes = [box for i, box in enumerate(thirdnewboxes) if i not in to_remove_green]
    fourthnewpreds = [pred for i, pred in enumerate(thirdnewpreds) if i not in to_remove_green]

    # 5th step:  REMOVE RED WITH MORE THAN 80% WITH ANOTHER RED
    to_remove_red = []  # List to store indices of red boxes with overlap

    for i, (box1, pred1) in enumerate(zip(fourthnewboxes, fourthnewpreds)):
        if pred1 == '1':  # Check if the box is red
            for j, (box2, pred2) in enumerate(zip(fourthnewboxes, fourthnewpreds)):
                if i != j and pred2 == '1':  # Check if the other box is also red
                    overlap_percentage = calculate_overlap(box1, box2)
                    if overlap_percentage >= 80:
                        # Remove the smaller red box
                        area1 = box1[2] * box1[3]
                        area2 = box2[2] * box2[3]
                        if area1 < area2:
                            to_remove_red.append(i)
                        else:
                            to_remove_red.append(j)

    # Remove red boxes with at least 80% overlap with another red box
    fifthnewboxes = [box for i, box in enumerate(fourthnewboxes) if i not in to_remove_red]
    fifthnewpreds = [pred for i, pred in enumerate(fourthnewpreds) if i not in to_remove_red]

    
    # End of optimization
    optimize_version = cv2.imread(filepath)

    # Draw the updated boxes on the image
    for box, pred in zip(fifthnewboxes, fifthnewpreds):
        x, y, w, h = box
        if pred == '0':
            cv2.rectangle(optimize_version, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif pred == '1':
            cv2.rectangle(optimize_version, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_img(img=optimize_version , filepath= filepath , title= 'Final_optimized_results')
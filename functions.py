from imports import *

# Function to plot images inline in the notebook
def show_img(img : np.ndarray , filepath : str , title=None ):
    # Check if the image is in BGR format and convert it to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.figure() 
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if filepath != "None":
        filename = os.path.basename(filepath)

        # Create the output directory path
        output_directory = './results/' + filename

        # Create the directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        # Save the figure inside the created directory
        output_path2 = os.path.join(output_directory, title)
        plt.savefig(output_path2)
    else:
        plt.show()

    plt.close()


# Preprocessing
def pre_processing_image(image : np.ndarray) -> (np.ndarray , np.ndarray):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find best threshold
    opt_value , _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return (blurred , opt_value)

def find_contours(image: np.ndarray) -> tuple:
    
    blurred , opt_value = pre_processing_image(image= image)

    # Find contours
    edges = cv2.Canny(blurred, opt_value * 0.3, opt_value)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def filter_contours_min_area(contours : tuple , area: int) -> list:

    boxcontainer = []
    # Loop over the contours
    for cnt in contours:
        # Get the minimum area rectangle that encloses the contour
        rect = cv2.minAreaRect(cnt)
        # Convert the rectangle to a box
        box = cv2.boxPoints(rect)

        box_area = cv2.contourArea(box)
        if box_area < 0.05 * area:        
            boxcontainer.append(box)

    # Remove boxes that are completely inside others
    filtered_boxcontainer = []
    for i, box1 in enumerate(boxcontainer):
        is_inside_other = False
        for j, box2 in enumerate(boxcontainer):
            if i != j and is_inside2(box1, box2):
                is_inside_other = True
                break
        if not is_inside_other:
            filtered_boxcontainer.append(box1)

    return filtered_boxcontainer

def cropping_and_sift_min_area_algorithm(image: np.ndarray , filtered_boxcontainer: list) -> list:
    new_images = []
    for box in filtered_boxcontainer:
        # Get the bounding box of the selected box
        x, y, w, h = cv2.boundingRect(box)
        cropped = image[y:y + h, x:x + w]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0: 
            grayversion = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
        new_images.append(grayversion)

    siftarrayelems22 = sift_computation(new_images)

    return siftarrayelems22


def first_filtering_layer(regions: list , area: int) -> list:
    # Initialize an empty set to store selected region proposals
    candidates = set()

    for r in regions:
        # Check if the current region's rectangle is already in candidates
        if r['rect'] in candidates:
            continue
        # Extract the coordinates and dimensions of the region's rectangle
        x, y, w, h = r['rect']
        # Avoid division by zero by checking if height or width is zero
        if h == 0 or w == 0:
            continue
        # Check the aspect ratio of the region (width / height and height / width)
        if w / h > 2 or h / w > 2:
            continue
        if w * h > 0.30 * area:
            continue
        # If all conditions are met, add the region's rectangle to candidates
        candidates.add(r['rect'])
        
    return candidates

def filter_inside_boxes(boxes):
    filtered_boxes = []

    for i, box1 in enumerate(boxes):
        is_inside_another = False

        for j, box2 in enumerate(boxes):
            if i != j and is_inside(box1, box2):
                is_inside_another = True
                break

        if not is_inside_another:
            filtered_boxes.append(box1)

    return filtered_boxes

def is_inside(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

def display_boxes_on_image(image: np.ndarray , boxes : list):
    output_image = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    show_img(img= output_image , filepath = "None")

def cropping_image(image: np.ndarray , boxes: list) -> list:
    cropped_images = []
    for (x,y,w,h) in boxes:
        grayversion = cv2.cvtColor(image[y:y + h, x:x + w],cv2.COLOR_BGR2GRAY)
        cropped_images.append(grayversion)
    return cropped_images

def sift_computation(images: list) -> list:
    sift = cv2.SIFT_create()
    siftarrayelems = []
    zeroarray = np.zeros(129)
    for elems in images:
        kp, des = sift.detectAndCompute(elems,None)
        if des is None:
            des = zeroarray
            siftarrayelems.append(des)
        else:
            stacked_array = np.stack((des), axis=0)
    
            # Compute the mean along the new axis (axis 0)
            mean_array = np.mean(stacked_array, axis=0)
            new_elem = np.array([len(des)])
            mean_array = np.append(mean_array, new_elem, axis=0)
            mean_array = mean_array / np.linalg.norm(mean_array)
            siftarrayelems.append(mean_array)
        
    return siftarrayelems

def results_with_pred(image: np.ndarray , boxes: list , predictions: np.ndarray):
    output = image.copy()
    
    for (x, y, w, h) , index in zip(boxes,predictions):
        if index == '0':
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif index == '1':
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    show_img(img=output , filepath="None")

def compute_best_MLP_with_CV(activation_f : str, parameters : dict, x_train : np.ndarray, y_train : np.ndarray) -> tuple:
    
    mlp = MLPClassifier(max_iter=1000, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1,activation = activation_f)

    grid_search = GridSearchCV(mlp, parameters, n_jobs = -1)
    grid_search.fit(x_train, y_train)

    best_param = grid_search.best_params_
    best_score = grid_search.best_score_
    all_scores = grid_search.cv_results_['mean_test_score']

    return best_param, best_score, all_scores

def automatic_min_area_threshold(contour_list):
    areas = [cv2.contourArea(cnt) for cnt in contour_list]
    threshold = np.percentile(areas, 70)
    return threshold

def filter_boxes_by_area(contour_list, index_list, min_area_threshold=None):
    if min_area_threshold is None:
        min_area_threshold = automatic_min_area_threshold(contour_list)

    valid_contours = []
    valid_indices = []

    for cnt, index in zip(contour_list, index_list):
        area = cv2.contourArea(cnt)

        if area > min_area_threshold:
            valid_contours.append(cnt)
            valid_indices.append(index)

    return valid_contours, valid_indices

def is_inside2(box1, box2):
    return cv2.pointPolygonTest(box2, (box1[0][0], box1[0][1]), False) > 0

def calculate_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    overlap_area = x_overlap * y_overlap
    area1 = w1 * h1
    area2 = w2 * h2

    overlap_percentage = (overlap_area / min(area1, area2)) * 100

    return overlap_percentage
    

def calculate_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    overlap_area = x_overlap * y_overlap
    area1 = w1 * h1
    area2 = w2 * h2

    overlap_percentage = (overlap_area / min(area1, area2)) * 100

    return overlap_percentage

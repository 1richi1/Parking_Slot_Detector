from imports import *
from functions import *
#### IMPORTING ####

def process_data():

    csv_file_path = './CNRPark+EXT.csv'

    # Create an empty list to store the dictionaries
    csv_data = []

    with open(csv_file_path, 'r') as file:
        # Create a CSV dictionary reader object
        csv_dict_reader = csv.DictReader(file)

        # Iterate through each row in the CSV file
        for row in csv_dict_reader:
            # Append each row (dictionary) to the list
            csv_data.append(row)

    # Getting data
    list_image = []
    y_list = []
    i=0
    for row in csv_data[:2000]:
        img = cv2.imread(row['image_url'])
        list_image.append(img)
        y_list.append(row['occupancy'])
    
    return list_image , y_list

from functions import *
from imports import *

from neural_network import neural_network
from selective_search_test import selective_search_test
from contours_search import contours_search
from connected_component_alg import connected_component_alg
from show_results import show_results

def main():

    model_filename = 'mlp_model.pkl'
    filepath = "./data/rainy_201512.jpg"  #Change this with your new image filepath
    with open(model_filename, 'rb') as file:
        mlp = pickle.load(file) 

    # Training
    #mlp = neural_network()  ## UNCOMMENT THIS LINE TO TRAIN THE MLP MODEL

    # 1st approach
    filtered_boxcontainer , pred2 , valid_cntlist , valid_indexlist = contours_search(mlp= mlp , filepath= filepath)
    
    # 2nd approach
    filtered_boxes , pred = selective_search_test(mlp = mlp , filepath= filepath)
    
    # 3rd approach
    regionprops_label_image , pred3 = connected_component_alg(mlp = mlp , filepath = filepath)

    # Results
    show_results(filepath=filepath,filtered_boxes=filtered_boxes , pred=pred , filtered_boxcontainer= filtered_boxcontainer , pred2= pred2 , regionprops_label_image= regionprops_label_image , pred3= pred3 , valid_cntlist= valid_cntlist , valid_indexlist= valid_indexlist)
    

if __name__ == "__main__":
    main()
from functions import *
from imports import *


def neural_network():
    sift = cv2.SIFT_create()
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
    for row in csv_data[:50000]:
        img = cv2.imread(row['image_url'])
        list_image.append(img)
        y_list.append(row['occupancy'])


    # TRAIN #

    ##SIFT DIRECTLY INTO IMAGE
    siftfullimage = []
    zeroarray = np.zeros(129)
    for image in list_image:
        if image is not None: 
            img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            _ , des = sift.detectAndCompute(img,None)
        if des is None:
            des = zeroarray
            siftfullimage.append(des)
        else:
            stacked_array = np.stack((des), axis=0)
            # Compute the mean along the new axis (axis 0)
            mean_array = np.mean(stacked_array, axis=0)
            new_elem = np.array([len(des)])
            mean_array = np.append(mean_array, new_elem, axis=0)
            mean_array = mean_array / np.linalg.norm(mean_array)
            siftfullimage.append(mean_array)
            

    m_t = 49000

    x_train, x_test, y_train, y_test = train_test_split(siftfullimage, y_list, train_size=m_t/len(y_list), random_state=1, stratify=y_list)

    # Here it is possible to test different architectures
    mlp_parameters = {'hidden_layer_sizes': [(10,)]}
    activation_functions = ['logistic']

    best_param_mlp, best_score_mlp, all_scores_mlp = {}, {}, {}

    for activation_f in activation_functions:
        best_param, best_score, all_scores = compute_best_MLP_with_CV(activation_f, mlp_parameters, x_train, y_train)
        best_param_mlp[activation_f] = best_param
        best_score_mlp[activation_f] = best_score
        all_scores_mlp[activation_f] = all_scores
    print(best_param_mlp, best_score_mlp , all_scores_mlp)

    best_activation_type = 'logistic'
    mlp_best_param = best_param_mlp['logistic']
    max_iter = 1000
    mlp = MLPClassifier(**mlp_best_param, max_iter=max_iter, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1,
                        learning_rate_init=.1,activation=best_activation_type, verbose=True)

    mlp.fit(x_train , y_train)

    # Save the trained model and other necessary data
    model_filename = 'mlp_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(mlp, file)

    # training_error = 1 - mlp.score(x_train , y_train)
    # test_error = 1 - mlp.score(x_test , y_test)

    # print ('\nRESULTS FOR BEST NN\n')
    # print ("Best NN training error: %f" % training_error)
    # print ("Best NN test error: %f" % test_error)

    # plt.plot(mlp.loss_curve_, label='Training Loss')
    # plt.title('Training loss MLP')
    # plt.xlabel('Iter'), plt.ylabel('Loss')

    # NN_prediction = mlp.predict(x_test)

    return mlp
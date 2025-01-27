# Import any necessary modules or libraries
import typing as tp
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

import sklearn
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


import pandas as pd
import selectivesearch
from itertools import chain

import scipy as sp
from scipy import stats
import csv

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from skimage import data

from skimage import io
from skimage import color
from skimage.transform import resize
import math
import numpy as np

import pickle
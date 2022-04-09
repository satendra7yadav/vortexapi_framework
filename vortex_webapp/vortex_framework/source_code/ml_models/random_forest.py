import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.datasets import make_classification

data = pd.read_csv("vortex_57_train.csv")
data_train = data[['CoordinateY', 'CoordinateZ','X_Component_Velocity_Dataset_V_X_N_1_1_0', 
                    'Y_Component_Velocity_Dataset_V_Y_N_1_1_0', 'Z_Component_Velocity_Dataset_V_Z_N_1_1_0', 
                    'Magnitude_Vorticity_Dataset_VORTICITY_MAG_N_1_1_0','Magnitude_Velocity_Dataset_V_MAG_N_1_1_0','Vortex']]

data_train
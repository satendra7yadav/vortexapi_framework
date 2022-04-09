from django.db import models
from .vortex_framework.source_code.ml_models.ensembled_model import *
from .vortex_framework.source_code.ml_models.ada_boost import *
from .vortex_framework.source_code.ml_models.lr_model import *
from .vortex_framework.source_code.ml_models.svm import *
from .vortex_framework.source_code.ml_models.xg_boost import *
from .vortex_framework.source_code.vision_framework.VortexDetection import _main

import matplotlib.pyplot as plt
import cv2
import os

# from vortex_framework.source_code.ensembled_model import *
# Create your models here.

root_dir = 'E:\\group project\\grp_project\\vortex_api_framework\\vortex_webapp\\'
model_dir = os.path.join(root_dir,'vortex_framework\\data\\trained_models\\')
image_dir = os.path.join(root_dir,'vortex_framework\\data\\Data\\')
os.makedirs(image_dir, exist_ok = True)
train_dataset_dir = os.path.join(root_dir,'vortex_framework\\data\\training_dataset_path\\')
os.makedirs(train_dataset_dir, exist_ok = True)
test_dataset_dir = os.path.join(root_dir,'vortex_framework\\data\\prediction_dataset_path\\')
os.makedirs(train_dataset_dir, exist_ok = True)
output_dir = os.path.join(root_dir,'vortex_framework\\data\\model_prediction\\')
os.makedirs(output_dir, exist_ok = True)

predict_model_func_dict = {
    'ensemble':ensembled_predict_model,
    'svm': svm_predict_model,
    'lr': lr_predict_model,
    'ada': ad_predict_model,
    'xgb': xgb_predict_model
}

train_model_func_dict = {
    'ensemble':ensembled_train_model,
    'svm': svm_train_model,
    'lr': lr_train_model,
    'ada': ad_train_model,
    'xgb': xgb_train_model
}

def plot_vortex_core(img, bbox_array, vortex_core, model, test_case, time_step):
    for bbox in bbox_array:
        x1, x2, y1, y2 = bbox
        img = cv2.rectangle(img, pt1 = (int(x1), int(y1)),  pt2 = (int(x2), int(y2)), color= (0,0,255))

    x = [(i[0]*675)/6.28 for i in vortex_core.values()]
    y = [(i[1]*675)/6.28 for i in vortex_core.values()] 
    print(len(x),len(y))
    size = 30 # size of marker
    # for i in vortex_core.values():
    #     img = cv2.circle(img, ((i[0]*675)/6.28,(i[1]*675)/6.28), radius=0, color=(255, 0, 0), thickness=-1)

    plt.imshow(img, cmap="gray") # plot image
    plt.scatter(x, y, size, c="r", marker="+") # plot markers
    output_path = os.path.join(output_dir,model,test_case)
    os.makedirs(output_path, exist_ok = True)
    plt.savefig(os.path.join(output_path,'time_step_'+str(time_step)+'_vortex_plot.png'))
    plt.close()
    image_path = os.path.join(output_path,'time_step_'+str(time_step)+'_vortex_plot.png')

    return image_path

def fetch_model_path(test_case,time_step,time_step_precision,total_time_steps,model):

    if (time_step%time_step_precision)==0:
        model_name = 'time_step_'+str(time_step)
        model_path = os.path.join(model_dir,model,test_case,model_name+'.pkl')
        return model_path

    time_step_value = (time_step//time_step_precision)*time_step_precision + time_step_precision

    if time_step_value>total_time_steps:
        model_name = 'time_step_'+str(total_time_steps)
        model_path = os.path.join(model_dir,model,test_case,model_name+'.pkl')
        return model_path
    else:
        model_name = 'time_step_'+str((time_step//time_step_precision)*time_step_precision + time_step_precision)
        model_path = os.path.join(model_dir,model,test_case,model_name+'.pkl')
        return model_path

def train_model(test_case,model,train_hyperparameters):

    train_hyperparameters['model_dir'] = model_dir
    train_hyperparameters['test_case']= test_case

    train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
    model_summary = train_model_func_dict.get(model)(train_dataset_path,train_hyperparameters)
    return model_summary

def predict_model(test_case,model,parameters):

    time_step = int(parameters.get('time_step'))
    time_step_precision = int(parameters.get('time_step_precision'))
    total_time_steps = int(parameters.get('total_time_steps'))
    test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
    model_path = fetch_model_path(test_case,time_step,time_step_precision,total_time_steps,model)
    image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
    no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
    vortex_core = predict_model_func_dict.get(model)(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
    img = cv2.imread(image_path)
    vortex_core['output_image_path'] = plot_vortex_core(img,bbox_array,vortex_core,model,test_case,time_step)
    return vortex_core
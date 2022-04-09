from ensembled_model.ensembled_model import *
from logistic_regression.lr_model import *
# from random_forest.random_forest import *
from ada_boost.ada_boost import *
from xg_boost.xg_boost import *
from svm.svm import *
from vision_framework.VortexDetection import _main
import glob
import argparse
import matplotlib.pyplot as plt1
import cv2
import os

# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print("root ",ROOT_DIR)
root_dir = 'E:\\group project\\grp_project\\'
model_dir = os.path.join(root_dir,'vortex_framework\\data\\trained_models\\')
image_dir = os.path.join(root_dir,'vortex_framework\\data\\Data\\')
os.makedirs(image_dir, exist_ok = True)
train_dataset_dir = os.path.join(root_dir,'vortex_framework\\data\\training_dataset_path\\')
os.makedirs(train_dataset_dir, exist_ok = True)
test_dataset_dir = os.path.join(root_dir,'vortex_framework\\data\\prediction_dataset_path\\')
os.makedirs(train_dataset_dir, exist_ok = True)
output_dir = os.path.join(root_dir,'vortex_framework\\data\\model_prediction\\')
os.makedirs(output_dir, exist_ok = True)

def plot_vortex_core(img, bbox_array, vortex_core, model, test_case, time_step):
    for bbox in bbox_array:
        x1, x2, y1, y2 = bbox
        img = cv2.rectangle(img, pt1 = (int(x1), int(y1)),  pt2 = (int(x2), int(y2)), color= (0,0,255))

    x = [(i[0]*675)/6.28 for i in vortex_core.values()]
    y = [(i[1]*675)/6.28 for i in vortex_core.values()]
    size = 50 # size of marker

    plt1.imshow(img, cmap="gray") # plot image
    plt1.scatter(x, y, size, c="r", marker="+") # plot markers
    output_path = os.path.join(output_dir,model,test_case)
    os.makedirs(output_path, exist_ok = True)
    plt1.savefig(os.path.join(output_path,'time_step_'+str(time_step)+'_vortex_plot.png'),bbox_inches='tight')
    plt1.show()
    # cv2.imshow("Bounding Boxes", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return img

def fetch_model_path(test_case,time_step,time_step_precision,model):

    model_name = 'time_step_'+str((time_step//time_step_precision)*time_step_precision + (time_step_precision//2))
    model_path = os.path.join(model_dir,model,test_case,model_name+'.pkl')
    return model_path

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-vortex', type=str, nargs=3, action="store", help='A required string positional argument to predict the model')


args = parser.parse_args()
print(args.vortex)
if args.vortex:
    if args.vortex[0]=='train':
        if args.vortex[1]=='ensemble':
            
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['model_dir'] = model_dir
            train_hyperparameters['test_case']='cube'
            train_hyperparameters['total_time_step']=int(args.vortex[2])
            train_hyperparameters['time_step_precision']= 20
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # ensemble model list
            train_hyperparameters['model_list']=['rfc','xgb']
            # random forest parameters
            train_hyperparameters['n_estimators']=200
            # xgboost parameters
            train_hyperparameters['learning_rate']=0.6
            train_hyperparameters['max_depth']=2
            train_hyperparameters['n_estimators']=200
            train_hyperparameters['subsample']=0.6
            train_hyperparameters['objective']='binary:logistic'

            test_case = train_hyperparameters['test_case']
            train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
            model_summary = ensembled_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='svm':
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['model_dir'] = model_dir
            train_hyperparameters['test_case']='cube'
            train_hyperparameters['total_time_step']=int(args.vortex[2])
            train_hyperparameters['time_step_precision']= 20
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # svm parameters
            train_hyperparameters['kernel']='poly'
            train_hyperparameters['C']=1
            train_hyperparameters['gamma']='auto'

            test_case = train_hyperparameters['test_case']
            train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
            model_summary = svm_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='lr':
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['model_dir'] = model_dir
            train_hyperparameters['test_case']='cube'
            train_hyperparameters['total_time_step']=int(args.vortex[2])
            train_hyperparameters['time_step_precision']= 20
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # svm parameters
            train_hyperparameters['penalty']='l2'
            train_hyperparameters['C']=40
            train_hyperparameters['solver']='lbfgs'
            
            test_case = train_hyperparameters['test_case']
            train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
            model_summary = lr_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='rf':
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['model_dir'] = model_dir
            train_hyperparameters['test_case']='cube'
            train_hyperparameters['total_time_step']=int(args.vortex[2])
            train_hyperparameters['time_step_precision']= 20
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # random forest parameters
            train_hyperparameters['n_estimators']=200

            test_case = train_hyperparameters['test_case']
            train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
            model_summary = rf_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='ada':
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['model_dir'] = model_dir
            train_hyperparameters['test_case']='cube'
            train_hyperparameters['total_time_step']=int(args.vortex[2])
            train_hyperparameters['time_step_precision']= 20
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # Ada boost parameters
            train_hyperparameters['max_depth']=5
            train_hyperparameters['learning_rate']=0.6
            train_hyperparameters['n_estimators']=200
            train_hyperparameters['algorithm']='SAMME'

            test_case = train_hyperparameters['test_case']
            train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
            model_summary = ad_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)

        elif args.vortex[1]=='xgb':
            train_hyperparameters= {}
            # dataset preparation parameters
            train_hyperparameters['model_dir'] = model_dir
            train_hyperparameters['test_case']='cube'
            train_hyperparameters['total_time_step']=int(args.vortex[2])
            train_hyperparameters['time_step_precision']= 20
            train_hyperparameters['test_size']=0.8
            train_hyperparameters['sampling_strategy']=0.8
            # xgboost parameters
            train_hyperparameters['learning_rate']=0.6
            train_hyperparameters['max_depth']=2
            train_hyperparameters['n_estimators']=200
            train_hyperparameters['subsample']=0.6
            train_hyperparameters['objective']='binary:logistic'

            test_case = train_hyperparameters['test_case']
            train_dataset_path = os.path.join(train_dataset_dir,test_case+'_vortex_trainset.csv')
            model_summary = xgb_train_model(train_dataset_path,train_hyperparameters)
            print(model_summary)
            
    elif args.vortex[0]=='predict':

        if args.vortex[1]=='ensemble':
            test_case = 'cube'
            time_step = int(args.vortex[2])
            time_step_precision = 20
            test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
            model_path = fetch_model_path(test_case,time_step,time_step_precision,str(args.vortex[1]))
            image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
            no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
            vortex_core = ensembled_predict_model(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
            print(vortex_core)
            img = cv2.imread(image_path)
            bbox_image=plot_vortex_core(img,bbox_array,vortex_core,args.vortex[1],test_case,time_step)

        elif args.vortex[1]=='svm':
            test_case = 'cube'
            time_step = int(args.vortex[2])
            time_step_precision = 20
            test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
            model_path = fetch_model_path(test_case,time_step,time_step_precision,str(args.vortex[1]))
            image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
            no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
            vortex_core = svm_predict_model(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
            print(vortex_core)
            img = cv2.imread(image_path)
            bbox_image=plot_vortex_core(img,bbox_array,vortex_core,args.vortex[1],test_case,time_step)

        elif args.vortex[1]=='lr':
            test_case = 'cube'
            time_step = int(args.vortex[2])
            time_step_precision = 20
            test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
            model_path = fetch_model_path(test_case,time_step,time_step_precision,str(args.vortex[1]))
            image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
            no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
            vortex_core = lr_predict_model(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
            print(vortex_core)
            img = cv2.imread(image_path)
            bbox_image=plot_vortex_core(img,bbox_array,vortex_core,args.vortex[1],test_case,time_step)

        elif args.vortex[1]=='rf':
            test_case = 'cube'
            time_step = int(args.vortex[2])
            time_step_precision = 20
            test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
            model_path = fetch_model_path(test_case,time_step,time_step_precision,str(args.vortex[1]))
            image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
            no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
            vortex_core = rf_predict_model(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
            print(vortex_core)
            img = cv2.imread(image_path)
            bbox_image=plot_vortex_core(img,bbox_array,vortex_core,args.vortex[1],test_case,time_step)

        elif args.vortex[1]=='ada':
            test_case = 'cube'
            time_step = int(args.vortex[2])
            time_step_precision = 20
            test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
            model_path = fetch_model_path(test_case,time_step,time_step_precision,str(args.vortex[1]))
            image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
            no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
            vortex_core = ada_predict_model(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
            print(vortex_core)
            img = cv2.imread(image_path)
            bbox_image=plot_vortex_core(img,bbox_array,vortex_core,args.vortex[1],test_case,time_step)

        elif args.vortex[1]=='xgb':
            test_case = 'cube'
            time_step = int(args.vortex[2])
            time_step_precision = 20
            test_dataset_path = os.path.join(test_dataset_dir,test_case+'_vortex_testset.csv')
            model_path = fetch_model_path(test_case,time_step,time_step_precision,str(args.vortex[1]))
            image_path = os.path.join(image_dir,test_case,'img'+str(time_step)+'.png')
            no_of_bounding_boxes, bbox_array=_main.CVMain(image_path)
            vortex_core = xgb_predict_model(test_dataset_path,model_path,no_of_bounding_boxes,time_step)
            print(vortex_core)
            img = cv2.imread(image_path)
            bbox_image=plot_vortex_core(img,bbox_array,vortex_core,args.vortex[1],test_case,time_step)
else:
    print('Incorrect argument')

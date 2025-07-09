import os 
import sys 
import pandas as pd 
import numpy as np 
# import dill
import pickle 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

import warnings
warnings.filterwarnings("ignore")

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try: 
        report={}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = models[model_name]
            param_grid = params.get(model_name, {}) 
            gs=GridSearchCV(model,param_grid,cv=3,n_jobs=-1)
            gs.fit(X_train,y_train)
            logging.info(f"Grid search CV on {model_name} Done!!")
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_test_pred=model.predict(X_test)
            y_train_pred=model.predict(X_train)
            
            test_model_score=r2_score(y_test,y_test_pred)
            train_model_score=r2_score(y_train,y_train_pred)
            report[list(models.keys())[i]]=test_model_score
            logging.info(f"{model_name} Trained Succesfully")
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
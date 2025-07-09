import sys 
import os
from dataclasses import dataclass


import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
class DataTransformation:
    def __init__(self):  
        self.data_transformation_obj=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            num_features=[ 'writing_score','reading_score']
            cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline=Pipeline( steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ])
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("oh_encoder",OneHotEncoder(handle_unknown="ignore")),
                ("scaler",StandardScaler(with_mean=False))
            ])
            logging.info(f"Categorical Features:{cat_features}")
            logging.info(f"Numerical Features: {num_features}")
            preprocessor=ColumnTransformer(
                [
                    ("Numerical Pipeline",num_pipeline,num_features),
                    ("Categorical Pipeline",cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessor_obj=self.get_data_transformation_object()
            target_column="math_score"
            numerical_columns=["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            output_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            output_feature_test_df=test_df[target_column]
            
            logging.info("Applying preprocessing on data both train and test data")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(output_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(output_feature_test_df)]
            logging.info(f"Saving the preprocessor object")
            save_object(file_path=self.data_transformation_obj.preprocessor_obj_file_path,obj=preprocessor_obj)
            return(train_arr,test_arr)
        except Exception as e:
            raise CustomException(e,sys)
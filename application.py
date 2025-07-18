from flask import Flask,request,render_template
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict_datapoint',methods=['GET','POST'])
def predict_datapoint():
    if(request.method=='GET'):
        return render_template('predict.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("After Prediction")
        print(f"Final Prediction: {round(results[0],2)}")
        return render_template('predict.html',results=results[0])
    
    
if __name__=="__main__":
    application.run(host="0.0.0.0")
    print("Server Running on http://localhost:3000")
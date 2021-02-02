# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:34:37 2021

@author: Rick
"""
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger
from sklearn.preprocessing import StandardScaler
sub=Flask(__name__)
Swagger(sub)
pickle_in=open('Client_subcription.pkl','rb')
subscribe_pre=pickle.load(pickle_in)
@sub.route('/')
def welcome():
    return "Welcome Everyone"
@sub.route('/predict',methods=['POST'])
def predic_subscription():

    """Let's predict the client will Subscribe or not
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    
    df_test=pd.read_csv(request.files.get("file"),sep=';')
    def data_cleaning(bank_loan):
        categorical_value=[feature for feature in bank_loan if bank_loan[feature].dtype =='O']
        for feature in categorical_value:
            temp=bank_loan[feature].unique()
            temp_df={i:j for j,i in enumerate(temp,0)}
            bank_loan[feature]=bank_loan[feature].map(temp_df)
        return bank_loan
    df_test=data_cleaning(df_test)
    
    scale=StandardScaler()
    scaled_data=scale.fit_transform(df_test)
    prediction=subscribe_pre.predict(scaled_data)
    return str(list(prediction))
if __name__=='__main__':
    sub.run()
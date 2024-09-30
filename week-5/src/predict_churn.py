import os as os
from pycaret.classification import  predict_model, load_model

def list_models( path ):
    dirs = os.listdir(path)
    return dirs

def make_prediction( path, model_name, data ):
    os.chdir(path)
    model = load_model(model_name)
    predictions = predict_model(model, data=data)
    return predictions



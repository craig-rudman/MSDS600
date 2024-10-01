import pandas as pd
from pycaret.classification import  predict_model, load_model

def make_prediction( data, model_name ):
    model = load_model(model_name)
    predictions = predict_model(model, data=data)
    # return prediction label and score
    result = pd.DataFrame(data=predictions, columns=['prediction_label', 'prediction_score'], copy=True)
    result = result.rename(columns={'prediction_label': 'Label', 'prediction_score': 'Score'})
    return result

import joblib
def predict(data):
    model = joblib.load('lstm_model.sav')
return model.predict(data)
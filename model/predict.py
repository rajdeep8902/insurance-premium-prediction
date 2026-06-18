import pickle
import pandas as pd

#import ML model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

#MLFlow   
MODEL_VERSION = "1.0.0"

#Get class labels from model (imp for matching probabs to class names)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):
    df = pd.DataFrame([user_input])
    
    #Predict the class
    predicted_class = model.predict(df)[0]
    
    #Get probabs for all classes
    probabs = model.predict_proba(df)[0]
    confidence = max(probabs)
    
    #Create mapping: {class_name: probab}
    class_probs = dict(zip(class_labels, map(lambda p: round(p, 4), probabs)))
    
    return {
        "predicted_category": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilites": class_probs
    }
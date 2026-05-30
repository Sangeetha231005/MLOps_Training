import pickle
import os
import numpy as np

__binary_model = None
__binary_tfidf = None

__crisis_model = None
__multi_tfidf = None
__label_encoder = None

__categories = None

def load_saved_artifacts():
    print("Loading hierarchical disaster classification artifacts... start")
    global __binary_model, __binary_tfidf
    global __crisis_model, __multi_tfidf, __label_encoder, __categories

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    bin_model_path = os.path.join(base_dir, 'disaster_detector.pkl')
    bin_tfidf_path = os.path.join(base_dir, 'disaster_vectorizer.pkl')
    multi_model_path = os.path.join(base_dir, 'crisis_model.pkl')
    multi_tfidf_path = os.path.join(base_dir, 'tfidf_vectorizer.pkl')
    encoder_path = os.path.join(base_dir, 'label_encoder.pkl')

    print(f"Loading binary model from: {bin_model_path}")
    with open(bin_model_path, 'rb') as f:
        __binary_model = pickle.load(f)
    print(f"Loading binary vectorizer from: {bin_tfidf_path}")
    with open(bin_tfidf_path, 'rb') as f:
        __binary_tfidf = pickle.load(f)

    print(f"Loading label encoder from: {encoder_path}")
    with open(encoder_path, 'rb') as f:
        __label_encoder = pickle.load(f)
    
        __categories = [str(cls) for cls in __label_encoder.classes_] + ["non-disaster"]

    print(f"Loading multi-class vectorizer from: {multi_tfidf_path}")
    with open(multi_tfidf_path, 'rb') as f:
        __multi_tfidf = pickle.load(f)

    print(f"Loading multi-class model from: {multi_model_path}")
    with open(multi_model_path, 'rb') as f:
        __crisis_model = pickle.load(f)

    print("Loading hierarchical disaster classification artifacts... done")

def get_categories():
    return __categories

def predict_disaster(text):
    if not text or not text.strip():
        return {
            "is_disaster": False,
            "binary_confidence": 100.0,
            "category": "non-disaster",
            "category_confidence": 100.0
        }

    vectorized_bin = __binary_tfidf.transform([text])
    
    is_disaster_bool = False
    bin_confidence = 1.0
    
    if hasattr(__binary_model, "predict_proba"):
        probabilities_bin = __binary_model.predict_proba(vectorized_bin)[0]
        disaster_proba = float(probabilities_bin[1])
        if disaster_proba >= 0.65:
            is_disaster_bool = True
            bin_confidence = disaster_proba
        else:
            is_disaster_bool = False
            bin_confidence = float(probabilities_bin[0])
    else:
        is_disaster = int(__binary_model.predict(vectorized_bin)[0])
        is_disaster_bool = (is_disaster == 1)
        bin_confidence = 1.0

    if not is_disaster_bool:
        return {
            "is_disaster": False,
            "binary_confidence": round(bin_confidence * 100, 2),
            "category": "non-disaster",
            "category_confidence": round(bin_confidence * 100, 2)
        }

    vectorized_multi = __multi_tfidf.transform([text])
    prediction = __crisis_model.predict(vectorized_multi)[0]
    predicted_label = __label_encoder.inverse_transform([prediction])[0]

    multi_confidence = 1.0
    if hasattr(__crisis_model, "predict_proba"):
        try:
            probabilities_multi = __crisis_model.predict_proba(vectorized_multi)[0]
            class_idx = list(__label_encoder.classes_).index(predicted_label)
            multi_confidence = float(probabilities_multi[class_idx])
        except Exception as e:
            print(f"Could not estimate multi-class probability: {e}")

    return {
        "is_disaster": True,
        "binary_confidence": round(bin_confidence * 100, 2),
        "category": str(predicted_label),
        "category_confidence": round(multi_confidence * 100, 2)
    }

if __name__ == '__main__':
    load_saved_artifacts()
    print("Loaded categories:", get_categories())
    
    test_1 = "I am watching a movie with my friends at home."
    res_1 = predict_disaster(test_1)
    print(f"Test 1: '{test_1}' -> {res_1}")

    test_2 = "Heavy flood waters are washing away houses, need emergency dispatch immediately!"
    res_2 = predict_disaster(test_2)
    print(f"Test 2: '{test_2}' -> {res_2}")

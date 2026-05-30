import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_hierarchical_model():
    print("==========================================================")
    print("STARTING DISASTER DETECTION & CLASSIFICATION TRAINING")
    print("==========================================================\n")
   
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_balanced = os.path.join(base_dir, 'final_updated_dataset_balanced.csv')

    
    if os.path.exists(csv_balanced):
        csv_path = csv_balanced
        print(f"Loading balanced dataset from: {csv_path}")
    else:
        print("Failed to load the dataset")
        return

    df = pd.read_csv(csv_path)
    df['clean_text'] = df['clean_text'].fillna('')
    
    print(f"Dataset shape: {df.shape}\n")
    
    print("----------------------------------------------------------")
    print("STEP 1: Training Binary Disaster Detector (Disaster vs. Non-Disaster)")
    print("----------------------------------------------------------")
    
    binary_tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9
    )
    
    X_bin = binary_tfidf.fit_transform(df['clean_text'])
    y_bin = df['disaster'].astype(int).values
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    binary_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    binary_model.fit(X_train_bin, y_train_bin)
    
    y_pred_bin = binary_model.predict(X_test_bin)
    bin_accuracy = accuracy_score(y_test_bin, y_pred_bin)
    print(f"Binary Detector Accuracy: {bin_accuracy:.4f}")
    print("\nBinary Classification Report:")
    print(classification_report(y_test_bin, y_pred_bin, target_names=['Non-Disaster (0)', 'Disaster (1)']))
    
    
    print("\n----------------------------------------------------------")
    print("STEP 2: Training Multi-Class Category Classifier (Emergency Types)")
    print("----------------------------------------------------------")
   
    df_disaster = df[df['disaster'] == True].copy()
    print(f"Disaster subset shape: {df_disaster.shape}")
    
    multi_tfidf = TfidfVectorizer(
        max_features=7000,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9
    )
    
    X_multi = multi_tfidf.fit_transform(df_disaster['clean_text'])
   
    encoder = LabelEncoder()
    y_multi = encoder.fit_transform(df_disaster['class_label'])
    categories = list(encoder.classes_)
    print(f"Target categories: {categories}")
    
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    multi_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    multi_model.fit(X_train_multi, y_train_multi)
    
    y_pred_multi = multi_model.predict(X_test_multi)
    multi_accuracy = accuracy_score(y_test_multi, y_pred_multi)
    print(f"Multi-Class Model Accuracy: {multi_accuracy:.4f}")
    print("\nMulti-Class Classification Report:")
    print(classification_report(y_test_multi, y_pred_multi, target_names=categories))
    
   
    print("\n----------------------------------------------------------")
    print("STEP 3: Saving Trained Models as Pickle Files")
    print("----------------------------------------------------------")
    
    bin_model_file = os.path.join(base_dir, 'disaster_detector.pkl')
    bin_tfidf_file = os.path.join(base_dir, 'disaster_vectorizer.pkl')
    multi_model_file = os.path.join(base_dir, 'crisis_model.pkl')
    multi_tfidf_file = os.path.join(base_dir, 'tfidf_vectorizer.pkl')
    encoder_file = os.path.join(base_dir, 'label_encoder.pkl')
    
    with open(bin_model_file, 'wb') as f:
        pickle.dump(binary_model, f)
    with open(bin_tfidf_file, 'wb') as f:
        pickle.dump(binary_tfidf, f)
    
    with open(multi_model_file, 'wb') as f:
        pickle.dump(multi_model, f)
    with open(multi_tfidf_file, 'wb') as f:
        pickle.dump(multi_tfidf, f)
    with open(encoder_file, 'wb') as f:
        pickle.dump(encoder, f)
        
    print(f"Saved: {os.path.basename(bin_model_file)}")
    print(f"Saved: {os.path.basename(bin_tfidf_file)}")
    print(f"Saved: {os.path.basename(multi_model_file)}")
    print(f"Saved: {os.path.basename(multi_tfidf_file)}")
    print(f"Saved: {os.path.basename(encoder_file)}")
    print("\nAll models trained and exported successfully! Hierarchical pipeline complete.")

if __name__ == '__main__':
    train_hierarchical_model()

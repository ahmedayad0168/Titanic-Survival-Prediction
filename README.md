# 🚢 Titanic Survival Prediction

This project predicts passenger survival on the **Titanic dataset** using Logistic Regression.  
It demonstrates data preprocessing, categorical encoding, model training, evaluation, and predictions.

## 📌 Steps in the Project
1. **Load & Clean Data**  
   - Dropped unnecessary columns (e.g., PassengerId, Name).  
   - Handled missing values (numeric → mean, categorical → "Missing").  

2. **Feature Engineering**  
   - Encoded categorical features with `LabelEncoder`.  
   - Combined numeric & categorical data.  

3. **Model Training**  
   - Trained a `LogisticRegression` classifier.  
   - Evaluated with train/test split accuracy.  

4. **Prediction**  
   - Applied model on test dataset.  
   - Saved results in `Survived.csv`.  

## ⚙️ Technologies
- Python  
- Pandas  
- Scikit-learn  

## 🚀 Run
```bash
pip install pandas scikit-learn
python titanic_prediction.py


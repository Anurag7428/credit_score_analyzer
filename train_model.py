import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Data Loading and Cleaning
file_path = r"C:/Users/anura/OneDrive/Desktop/credit_score_mode/test.csv"
def load_and_clean_data(file_path):
    df = pd.read_csv(r"C:/Users/anura/OneDrive/Desktop/credit_score_mode/test.csv")
    

    # Handle missing values and special characters
    df.replace(['NA', 'NM', '#F%$D@*&8', '!@9#%8', '__10000__', ''], np.nan, inplace=True)

    # Convert Credit_History_Age to months
    def convert_age(x):
        if pd.isna(x):
            return np.nan
        years = 0
        months = 0
        if 'Year' in str(x):
            years = int(str(x).split('Year')[0].strip())
        if 'Month' in str(x):
            months_part = str(x).split('and')[-1] if 'and' in str(x) else str(x)
            months = int(months_part.split('Month')[0].strip())
        return years * 12 + months

    df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_age)

    # Clean all numeric columns
    numeric_columns = ['Age', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 
                       'Outstanding_Debt', 'Changed_Credit_Limit', 'Amount_invested_monthly',
                       'Monthly_Inhand_Salary', 'Annual_Income']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

    df.loc[df['Annual_Income'] > 1000000, 'Annual_Income'] = np.nan

    return df

# Feature Engineering
def create_features(df):
    df['Debt_to_Income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1)
    df['Utilization_Ratio'] = df['Credit_Utilization_Ratio'] / 100
    df['EMI_to_Income'] = df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1)
    df['Monthly_Savings'] = df['Monthly_Inhand_Salary'] - df['Total_EMI_per_month'] - df['Amount_invested_monthly']
    df['Num_Loan_Types'] = df['Type_of_Loan'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['Spent_Value'] = df['Payment_Behaviour'].apply(lambda x: str(x).split('_')[0] if pd.notna(x) else 'Unknown')
    df['Payment_Size'] = df['Payment_Behaviour'].apply(lambda x: str(x).split('_')[-1] if pd.notna(x) else 'Unknown')
    df['Months_With_Credit'] = df.groupby('Customer_ID')['Month'].transform('count')
    df['Late_Payment'] = (df['Num_of_Delayed_Payment'] > 0).astype(int)
    return df

#  Data Preparation
def prepare_data(df):
    features = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_of_Loan', 'Total_EMI_per_month', 'Outstanding_Debt',
        'Credit_Mix', 'Payment_of_Min_Amount'
    ]
    target = 'Late_Payment'

    model_df = df[features + [target]].copy()

    # Encode categoricals
    cat_cols = ['Credit_Mix', 'Payment_of_Min_Amount']
    for col in cat_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))

    # Handle missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_cols = [col for col in features if col not in cat_cols]

    model_df[num_cols] = model_df[num_cols].apply(pd.to_numeric, errors='coerce')
    model_df[num_cols] = num_imputer.fit_transform(model_df[num_cols])
    model_df[cat_cols] = cat_imputer.fit_transform(model_df[cat_cols])

    # BALANCE CLASSES HERE:
    majority = model_df[model_df[target] == 0]
    minority = model_df[model_df[target] == 1]
    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=42)
    model_df = pd.concat([majority, minority_upsampled])

    # Scale numeric columns
    scaler = StandardScaler()
    model_df[num_cols] = scaler.fit_transform(model_df[num_cols])

    X = model_df[features]
    y = model_df[target]

    return X, y, scaler, features


#  Model Training
def train_and_save_model(X, y, scaler):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    # Feature importance
    plt.figure(figsize=(12, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances")
    sns.barplot(x=importances[indices], y=np.array(X.columns)[indices])
    plt.tight_layout()
    plt.show()

    with open('models/credit_scoring_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/credit_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Model and scaler saved successfully!")
    return model

# Main execution
if __name__ == '__main__':
    print("Loading and cleaning data...")
    file_path = r"C:/Users/anura/OneDrive/Desktop/credit_score_mode/test.csv"
    df = load_and_clean_data(file_path)

    print("Creating features...")
    df = create_features(df)

    print("Preparing data for modeling...")
    X, y, scaler, features = prepare_data(df)

    print("Training model...")
    model = train_and_save_model(X, y, scaler)

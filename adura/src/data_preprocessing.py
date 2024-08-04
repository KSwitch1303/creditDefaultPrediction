import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    label_encoder = LabelEncoder()
    df['Loan Repayment History'] = label_encoder.fit_transform(df['Loan Repayment History'])
    df['Employment Status'] = label_encoder.fit_transform(df['Employment Status'])
    df['Credit Worthiness'] = label_encoder.fit_transform(df['Credit Worthiness'])

    X = df.drop('Credit Worthiness', axis=1)
    y = df['Credit Worthiness']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, label_encoder

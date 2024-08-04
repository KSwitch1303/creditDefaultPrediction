import pandas as pd

def predict_credit_worthiness(model, scaler, label_encoder, account_balance, transaction_history, loan_repayment_history,
                              credit_card_usage, number_of_loans, income_level, age, employment_status):
    input_data = pd.DataFrame({
        'Account Balance': [account_balance],
        'Transaction History': [transaction_history],
        'Loan Repayment History': [label_encoder.transform([loan_repayment_history])[0]],
        'Credit Card Usage (%)': [credit_card_usage],
        'Number of Loans': [number_of_loans],
        'Income Level': [income_level],
        'Age': [age],
        'Employment Status': [label_encoder.transform([employment_status])[0]]
    })
    
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    return 'Good' if prediction[0] == 1 else 'Bad'

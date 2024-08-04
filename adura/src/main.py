from src.data_preprocessing import preprocess_data
from src.model_training import train_model, evaluate_model
from src.prediction import predict_credit_worthiness

# Preprocess data
X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data('data/synthetic_data.csv')

# Train model
rf_model = train_model(X_train, y_train)

# Evaluate model
accuracy, precision, recall, f1 = evaluate_model(rf_model, X_test, y_test)
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Predict credit worthiness for a new user
credit_worthiness = predict_credit_worthiness(rf_model, scaler, label_encoder, 50000, 20, 'Yes', 30, 1, 80000, 35, 'Employed')
print(f'Predicted Credit Worthiness: {credit_worthiness}')

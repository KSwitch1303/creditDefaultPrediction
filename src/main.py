import logging
from preprocess import load_data, preprocess_data
from model import create_ensemble_model, train_model, evaluate_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting main process...")
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    
    logger.info("Loading data...")
    df = load_data(data_url)
    
    logger.info("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    logger.info("Creating ensemble model...")
    model = create_ensemble_model()
    
    logger.info("Training ensemble model...")
    trained_model = train_model(model, X_train, y_train)
    
    logger.info("Evaluating ensemble model...")
    evaluate_model(trained_model, X_test, y_test)
    logger.info("Main process complete.")

if __name__ == "__main__":
    main()

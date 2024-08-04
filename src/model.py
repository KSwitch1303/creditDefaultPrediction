import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_ensemble_model():
    logger.info("Creating ensemble model...")
    
    logger.info("Initializing individual classifiers...")
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = GradientBoostingClassifier(n_estimators=100)
    
    logger.info("Combining classifiers into a VotingClassifier...")
    ensemble_model = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='hard')
    
    logger.info("Ensemble model created successfully.")
    return ensemble_model

def train_model(model, X_train, y_train):
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model...")
    accuracy = model.score(X_test, y_test)
    logger.info(f'Model Accuracy: {accuracy}')
    print(f'Model Accuracy: {accuracy}')

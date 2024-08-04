import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(url):
    logger.info("Loading data from URL...")
    df = pd.read_excel(url, skiprows=1)
    logger.info("Data loaded successfully.")
    return df

def preprocess_data(df):
    logger.info("Preprocessing data...")
    
    logger.info("Dropping unnecessary columns...")
    X = df.drop(columns=['ID', 'default payment next month'])
    y = df['default payment next month']
    
    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    logger.info("Standardizing the features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info("Data preprocessing complete.")
    return X_train, X_test, y_train, y_test

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
data = {
    'Account Balance': np.random.uniform(0, 100000, n_samples),
    'Transaction History': np.random.randint(0, 100, n_samples),
    'Loan Repayment History': np.random.choice(['Yes', 'No'], n_samples),
    'Credit Card Usage (%)': np.random.uniform(0, 100, n_samples),
    'Number of Loans': np.random.randint(0, 5, n_samples),
    'Income Level': np.random.uniform(20000, 200000, n_samples),
    'Age': np.random.randint(18, 70, n_samples),
    'Employment Status': np.random.choice(['Employed', 'Unemployed'], n_samples),
    'Credit Worthiness': np.random.choice(['Good', 'Bad'], n_samples)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
df.head()

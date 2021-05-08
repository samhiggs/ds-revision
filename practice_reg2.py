import logging
from time import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
df = pd.read_csv("data/train.csv")
df = df.sample(10000, random_state=42)


##### PREPROCESSING ######
print("Training Regression Model")
logging.info("1. Setting index and renaming Annual Premium col")

# Set index
df.set_index("id", inplace=True)

# Set Response
df.rename(columns={"Annual_Premium": "dependent"}, inplace=True)

logging.info("2. deduplication")
# Check if duplicate IDs exist
if not df.index.is_unique:
    # Deduplicate id
    df.loc[~df.index.duplicated(), :]

logging.info("3. Remove missing Responses")
# Remove any missing Responses
df.dropna(subset=["dependent"], inplace=True)

# We need to adjust a few of the variables which are currently numeric and need to be categorical
cols_to_cat = ["Driving_License", "Previously_Insured", "Policy_Sales_Channel", "Region_Code", "Response"]
logging.info(f"4. Converting {cols_to_cat} to categorical columns")
df[cols_to_cat] = df[cols_to_cat].astype("category")

cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols = df.select_dtypes(exclude=["category", "object"]).columns.tolist()

logging.info(f"5. Getting categorical and numerical cols \
    {len(cat_cols) + len(num_cols)} should equal {df.shape[1]}")
logging.info(f"6. Imputing numerical cols")
# Impute numerical with mean
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

logging.info(f"7. Imputing categorical cols")
# Impute categorical with most frequent
for col in cat_cols:
    df[col].fillna(df[col].mode(), inplace=True)

logging.info(f"8. Normalizing numerical cols")
# Normalize numerical cols
num_cols.remove("dependent")
df[num_cols] = df[num_cols].apply(lambda x: x/x.max(), axis=0)

logging.info(f"9. OneHot Encoding categorical cols")
# Onehot encode categorical cols (excluding Response)
df = pd.get_dummies(df, columns=cat_cols)

##### MODEL #####
print("\n\n~~~~~~~~~Setting up and training model~~~~~~~~~\n")
logging.debug(df.columns)
logging.debug(df.head())
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("dependent", axis=1),
    df.dependent,
    test_size=0.2,
    random_state=42
)

reg = LinearRegression()
logging.info("Training model")
t1 = time()
reg.fit(X_train, y_train)
t2 = time()
logging.info(f"Training took {t2-t1:.2f} seconds")

t1 = time()
y_test_pred = reg.predict(X_test)
t2 = time()
print(f"Predictions took {t2-t1:.2f} seconds")

print("Explaining the Regressor\n")
print(f"Regression intercept: {reg.intercept_}")
# print(f"Coefficients")
# print(reg.coef_)

print("\n\n~~~~~~~~Evaluation~~~~~~~~~~~\n")
print(f"Annual Premium mean: {df.dependent.mean()} std_Dev: {df.dependent.std()}")
print(f"Mean Absolute Error: {round(mean_absolute_error(y_test, y_test_pred),1)}")
print(f"Mean Squared Error: {round(mean_squared_error(y_test, y_test_pred),1)}")
print(f"Root Mean Squared Error: {round(np.sqrt(mean_squared_error(y_test, y_test_pred)),1)}")




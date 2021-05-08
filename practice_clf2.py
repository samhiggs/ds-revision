import logging
from time import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
df = pd.read_csv("data/train.csv")
df = df.sample(10000, random_state=42)


##### PREPROCESSING ######
logging.info("1. Setting index and renaming label col")

# Set index
df.set_index("id", inplace=True)

# Set label
df.rename(columns={"Response": "label"}, inplace=True)

logging.info("2. deduplication")
# Check if duplicate IDs exist
if not df.index.is_unique:
    # Deduplicate id
    df.loc[~df.index.duplicated(), :]

logging.info("3. Remove missing labels")
# Remove any missing labels
df.dropna(subset=["label"], inplace=True)

# We need to adjust a few of the variables which are currently numeric and need to be categorical
cols_to_cat = ["Driving_License", "Previously_Insured", "Policy_Sales_Channel", "Region_Code", "label"]
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
df[num_cols] = df[num_cols].apply(lambda x: x/x.max(), axis=0)

logging.info(f"9. OneHot Encoding categorical cols")
# Onehot encode categorical cols (excluding label)
cat_cols.remove("label")
df = pd.get_dummies(df, columns=cat_cols)

##### MODEL #####
print("\n\n~~~~~~~~~Setting up and training model~~~~~~~~~\n")
logging.debug(df.columns)
logging.debug(df.head())
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("label", axis=1),
    df.label,
    test_size=0.2,
    random_state=42,
    stratify=df.label
)

clf = RandomForestClassifier()
logging.info("Training model")
t1 = time()
clf.fit(X_train, y_train)

t2 = time()
logging.info(f"Training took {t2-t1:.2f} seconds")

t1 = time()
y_test_pred = clf.predict(X_test)
t2 = time()
print(f"Predictions took {t2-t1:.2f} seconds")

print(f"Accuracy_score: {round(accuracy_score(y_test, y_test_pred),3)}\n")

print(f"Confusion matrix:\n")
print(f"{pd.DataFrame(confusion_matrix(y_test, y_test_pred))}\n")

print(classification_report(y_test, y_test_pred))

t1 = time()
logit = LogisticRegression()
logit.fit(X_train, y_train)
t2 = time()
logging.info(f"Training took {t2-t1:.2f} seconds")

t1 = time()
y_test_pred_logit = clf.predict(X_test)
t2 = time()
print(f"Predictions took {t2-t1:.2f} seconds")

print(f"Accuracy_score: {round(accuracy_score(y_test, y_test_pred_logit),3)}\n")

print(f"Confusion matrix:\n")
print(f"{pd.DataFrame(confusion_matrix(y_test, y_test_pred_logit))}\n")

print(classification_report(y_test, y_test_pred_logit))




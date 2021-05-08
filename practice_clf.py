import logging
from time import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import \
    confusion_matrix, \
    accuracy_score,\
    classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

def evaluate(clf, X_test, y_test):
    """
    Simple function to predict, evaluate and display
    the effectiveness of the classifier
    Args:
        clf (sklearn.Estimator): Trained scikit-learn model
        X_test (pd.DataFrame): test features
        y_test (pd.Series): The actual test labels
    Returns:
        y_test_pred (pd.Series): Predictions from model
    """
    t1 = time()
    y_test_pred = clf.predict(X_test)
    t2 = time()
    print(f"{len(y_test_pred)} Predictions took {t2-t1:.2f} seconds to classify")
    # Display evaluation
    print(f"Accuracy_score: {round(accuracy_score(y_test, y_test_pred),3)}\n")
    print(f"Confusion matrix:\n")
    print(f"{pd.DataFrame(confusion_matrix(y_test, y_test_pred))}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    return y_test_pred


logging.basicConfig(level=logging.INFO)
df = pd.read_csv("data/dataset_31_credit-g.csv")

##### PREPROCESSING ######
logging.info("1. Setting index and renaming label col")

# Set index
# df.set_index("id", inplace=True)

# Set label
df.rename(columns={"class": "label"}, inplace=True)

# Show the balance of the labels
print(f"Distribution of labels\n{df.label.value_counts()}\n")


logging.info("2. deduplication")
# Check if duplicate IDs exist
if not df.index.is_unique:
    # Deduplicate id
    df.loc[~df.index.duplicated(), :]

logging.info("3. Remove missing labels")
# Remove any missing labels
df.dropna(subset=["label"], inplace=True)

# We need to adjust a few of the variables which are currently numeric
# and need to be categorical
cols_to_cat = [
    "installment_commitment"
]
logging.info(f"4. Converting {cols_to_cat} to categorical columns")
df[cols_to_cat] = df[cols_to_cat].astype("category")

cat_cols = df.select_dtypes(
    include=["category", "object"]).columns.tolist()
num_cols = df.select_dtypes(
    exclude=["category", "object"]).columns.tolist()

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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("label", axis=1),
    df.label,
    test_size=0.2,
    random_state=42,
    stratify=df.label
)

# Train base model
# Simplest model to create is a logistic regression as it is easy to
# explain, fast to run and easy to embed in most enterprise settings.
print("Training Logistic Regression Model")
t1 = time()
logit = LogisticRegression()
logit.fit(X_train, y_train)
t2 = time()
logging.info(f"Logistic Regression Training took {t2-t1:.2f} seconds")

# Evaluate logistic regression
print("~~~~~~~~~~~START Logistic Regression Evaluation~~~~~~~~~~~~~\n")
evaluate(logit, X_test, y_test)
print("~~~~~~~~~~~END Logistic Regression Evaluation~~~~~~~~~~~~~\n")

# Train more complex model
# A more complex model, which requires more complexity to explain is
# a random forest model which can handle encoded variables much better
# and can be trained with many more hyperparameters

# Define hyperparameters thinking about avoiding overfitting
rs_params = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

# Initialise model
rf = RandomForestClassifier()

# Initialise cross validation with random search for optimal params
clf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rs_params,
    n_iter = 3,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

logging.info("Training Random Forest model")
t1 = time()
clf.fit(X_train, y_train)
t2 = time()
logging.info(f"Training took {t2-t1:.2f} seconds")

print("~~~~~~~~~~~START Random Forest Evaluation~~~~~~~~~~~~~\n")
evaluate(clf.best_estimator_, X_test, y_test)
print("~~~~~~~~~~~END Random Forest Evaluation~~~~~~~~~~~~~\n")



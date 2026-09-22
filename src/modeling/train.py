"""Train a simple first model.

Run it from the project folder, after `src.dataset`, with:

    python -m src.modeling.train

This is a CLASS EXAMPLE of how a project is organized. It is not an acceptable
baseline for the course project: five columns, no validation slice, no
treatment of class imbalance, no threshold. Your group chooses and justifies
its own features, in its own data audit.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CLEAN_CSV, MODEL_FILE, SEED
from src.features import make_features


def train():
    """Fit a logistic regression on the train rows and score it on the test rows.

    The data already has a `split` column that says which rows are for
    training (loans from 2014 to mid 2016) and which are for testing (later
    loans), so we use it and do not split again. The numbers are scaled first,
    inside the same pipeline, so the saved model does the scaling by itself
    when it predicts.

    Returns:
        float: The ROC AUC on the test rows. 0.5 is a coin toss, 1.0 is perfect.
        ROC AUC is the same regardless of which outcome is called positive.

    Raises:
        FileNotFoundError: If `src.dataset` has not been run yet.
    """
    df = pd.read_csv(CLEAN_CSV)
    X_train, y_train = make_features(df[df["split"] == "train"])
    X_test, y_test = make_features(df[df["split"] == "test"])

    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    joblib.dump(model, MODEL_FILE)
    return auc


if __name__ == "__main__":
    print(f"ROC AUC on the test rows: {train():.3f}")
    print(f"Model saved to {MODEL_FILE}")

import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # data analysis
    import pandas as pd
    import numpy as np
    import random as rnd

    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt

    # imputer (Missing Data Handling)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    # from sklearn.pipeline import Pipeline
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from imblearn.over_sampling import SMOTE
    from sklearn.feature_selection import RFECV

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    from sklearn.metrics import make_scorer, balanced_accuracy_score
    from sklearn.utils.class_weight import compute_class_weight

    from sklearn.metrics import (classification_report, roc_curve, roc_auc_score,
                                 confusion_matrix, accuracy_score, f1_score, precision_score, 
                                 recall_score)
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import auc
    return (
        LGBMClassifier,
        LabelEncoder,
        Pipeline,
        SMOTE,
        StandardScaler,
        StratifiedKFold,
        accuracy_score,
        classification_report,
        confusion_matrix,
        cross_validate,
        np,
        pd,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _():
    # os
    import os
    return


@app.cell
def _(pd):
    # Pd Display
    def pdisplay(df):
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(df)
        else:
            return df
    return


@app.cell
def _(mo):
    mo.md(r"""# Import Data""")
    return


@app.cell
def _(mo):
    train_df_path = mo.notebook_location() /'..' / 'Data' / 'Supplementary Information File 1 (SIF-1).xlsx'
    test_df_path = mo.notebook_location() /'..'/ 'Data'/ 'Supplementary Information File 2 (SIF-2).xlsx'
    return test_df_path, train_df_path


@app.cell
def _(pd, test_df_path, train_df_path):
    train_df = pd.read_excel(train_df_path)
    test_df = pd.read_excel(test_df_path)
    return test_df, train_df


@app.cell
def _(train_df):
    train_df.head()
    return


@app.cell
def _(test_df):
    test_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""# Data Analysis""")
    return


@app.cell
def _(train_df):
    train_df.dtypes
    return


@app.cell
def _(train_df):
    train_df['Lowest distortion'].value_counts()
    return


@app.cell
def _():
    # no_ld_idx = train_df['Lowest distortion'] == '-'
    # train_df[no_ld_idx].shape[0]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    - 53 values are not specified in lowest distortion.
    - they are replaced by `-`
    """
    )
    return


@app.cell
def _(train_df):
    train_df['v(A)'].value_counts()
    return


@app.cell
def _():
    # vA_idx = train_df['v(A)']
    # no_vA_idx = (vA_idx == 'not balanced') | (vA_idx == 'element not in BV')
    # mo.md(f"Unavailable v(A) - {train_df[no_vA_idx].shape[0]}")

    return


@app.cell
def _():
    # # not (no_Lowest_distion) and (no_vA_idx)
    # only_no_vA_idx = ~(no_ld_idx) & no_vA_idx
    # train_df[only_no_vA_idx].shape[0]
    return


@app.cell
def _(mo):
    mo.md(r"""- total unavailable data 3063""")
    return


@app.cell
def _(train_df):
    train_df['τ'].value_counts
    return


@app.cell
def _(train_df):
    train_df.isnull().sum()
    return


@app.cell
def _(train_df):
    train_df.describe()
    return


@app.cell
def _(train_df):
    train_df.describe(include=['O'])
    return


@app.cell
def _(mo):
    mo.md(r"""# Missing Data Handling""")
    return


@app.cell
def _(train_df):
    train_df.columns
    return


@app.cell
def _(train_df):
    no_ld_idx = train_df['Lowest distortion'] != '-'

    vA_idx = train_df['v(A)']
    no_vA_idx = (vA_idx != 'not balanced') & (vA_idx != 'element not in BV')

    no_T_idx = train_df['τ'] != '-'

    tG_idx = train_df['tG']
    no_tG_idx = (0.82 < tG_idx) & (tG_idx < 1.10)

    u_idx = train_df['μ']
    no_u_idx = (0.414 < u_idx) & (u_idx < 0.732)

    final_idx = no_ld_idx & no_vA_idx & no_T_idx & no_tG_idx & no_u_idx
    train_df[final_idx]
    return (final_idx,)


@app.cell
def _(LabelEncoder, final_idx, pd, train_df):
    vA = train_df[final_idx]['v(A)']
    feature_drop = ["S.No" , "Compound", "A", "B", "In literature", "v(B)", 'r(BVI)(Å)']
    train_df_final = train_df[final_idx].drop(feature_drop, axis=1)

    # T
    train_df_final['τ'] = train_df_final['τ'].apply(pd.to_numeric, errors="coerce")

    # One Hot Encoding
    train_df_final = train_df_final.drop(['v(A)'], axis=1)
    train_df_final['v(A)_1'] = vA == 1
    train_df_final['v(A)_2'] = vA == 2
    train_df_final['v(A)_3'] = vA == 3

    # Label encoding
    ld = train_df_final['Lowest distortion']
    # train_df_final = train_df_final.drop(['Lowest distortion'], axis=1)
    # create a LabelEncoder instance
    le = LabelEncoder()
    # fit and transform the 'A' column
    train_df_final['Lowest distortion'] = le.fit_transform(train_df_final['Lowest distortion'])



    train_df_final
    return le, train_df_final


@app.cell
def _(plt, sns, train_df_final):
    _corr = train_df_final.corr(method="spearman")
    plt.figure(figsize=(14,6))
    sns.heatmap(_corr, annot=True)
    return


@app.cell
def _(mo):
    mo.md(r"""# Model Train""")
    return


@app.cell
def _(SMOTE, train_df_final, train_test_split):
    X = train_df_final.drop(['Lowest distortion'], axis=1)
    y = train_df_final['Lowest distortion']
    smote = SMOTE(random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=100)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X, X_test, X_train, y, y_test, y_train


@app.cell
def _(LGBMClassifier, X_test, X_train, y_train):

    from sklearn.model_selection import cross_val_score
    # from sklearn.metrics import accuracy_score

    # Define the hyperparameters
    params = {'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05}

    # Create a 5-fold cross-validation object
    cv = 5

    # Create a LightGBM classifier
    # clf = lgb.LGBMClassifier(**params)

    lgbm = LGBMClassifier(
        # force_col_wise=True,
        objective="multiclass", 
        # n_estimators=100, 
        learning_rate=0.01
    )
    lgbm.fit(X_train, y_train )

    y_pred = lgbm.predict(X_test)

    # Perform 5-fold cross-validation
    # scores = cross_val_score(lgbm, X, y, cv=cv, scoring='accuracy')
    # # Print the average accuracy score
    # mo.vstack([
    #     mo.md(f'Average accuracy score: {scores.mean()*100:0.2f} %'),
    #     mo.md(f'Standard Deviation: {scores.std():0.3f}')
    # ])
    return lgbm, y_pred


@app.cell
def _(classification_report, le, y_pred, y_test):
    # Generate raw classification report dictionary
    report_dict = classification_report(
        y_test, y_pred, 
        target_names=le.classes_,
        output_dict=True,
        digits=3
    )
    # report_dict
    return


@app.cell
def _(confusion_matrix, plt, sns, y_pred, y_test):
    # from sklearn.metrics import confusion_matrix

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='d')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.gca()
    return


@app.cell
def _(
    Pipeline,
    StandardScaler,
    StratifiedKFold,
    X,
    cross_validate,
    lgbm,
    np,
    pd,
    y,
):
    sk = StratifiedKFold(shuffle = True, random_state = 100, n_splits= 5)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        # ('smote' , SMOTE(random_state=42) ),
        ('LightGBM', lgbm)
    ])

    result = cross_validate(pipe, X, y, cv = sk, n_jobs= -1)
    scores_cv:np.array = pd.DataFrame(result)['test_score'].values
    _result = [
        {'Accurecy': scores_cv.mean().round(4)*100,
        'SD': scores_cv.std().round(3) },
    ]
    pd.DataFrame(_result)
    return


@app.cell(hide_code=True)
def _(classification_report, le, mo, pd, y_pred, y_test):
    def classification_report_to_dataframe(y_true, y_pred, class_names=None, digits=2, title="Classification Report", title_level=1):
        """
        Convert scikit-learn classification report to pandas DataFrame

        Parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names in order
        digits (int): Number of decimal places for metrics

        Returns:
        tuple: (metrics_df, summary_df) - Metrics by class and summary statistics
        """
        # Generate raw classification report dictionary
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True,
            digits=digits
        )

        # Extract class-wise metrics and support
        metrics_df = pd.DataFrame(report_dict).transpose().reset_index()
        metrics_df = metrics_df.rename(columns={'index': 'class'})

        # Separate class metrics from summary statistics
        summary_mask = metrics_df['class'].isin(['accuracy', 'macro avg', 'weighted avg'])
        summary_df = metrics_df[summary_mask].copy()
        metrics_df = metrics_df[~summary_mask].copy()

        # Clean up formatting
        for col in ['precision', 'recall', 'f1-score']:
            metrics_df[col] = metrics_df[col].apply(
                lambda x: f"{x:.{digits}f}" if not pd.isna(x) else ""
            )
            summary_df[col] = summary_df[col].apply(
                lambda x: f"{x:.{digits}f}" if not pd.isna(x) else ""
            )

        # Add total support count
        total_support = metrics_df['support'].sum()
        summary_df.loc[summary_df['class'] == 'accuracy', 'support'] = total_support

        report = mo.vstack([
        mo.md(f"{'#'*title_level} {title}"),
        mo.md("## Class-wise Metrics:"),
        mo.md(metrics_df.to_markdown(index=False)),
        mo.md("## Summary Statistics:"),
        mo.md(summary_df.to_markdown(index=False))
        ])
        return report

    classification_report_to_dataframe(
        y_test, y_pred,
        class_names=le.classes_,
        digits=3
    )
    return


@app.cell
def _(accuracy_score, y_pred, y_test):
    accuracy_score(y_test, y_pred)
    return


@app.cell
def _(LGBMClassifier, SMOTE, StratifiedKFold, X, np, y):
    def _():
        RANDOM_STATE = 42

        # Initialize LightGBM model
        lgb_model = LGBMClassifier(
            objective='multiclass',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.1
        )

        # Stratified K-Fold Cross Validation with SMOTE
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        smote = SMOTE(random_state=RANDOM_STATE)

        cv_scores = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Apply SMOTE only to training data
            X_res, y_res = smote.fit_resample(X_train, y_train)

            # Train model
            lgb_model.fit(X_res, y_res)

            # Evaluate
            score = lgb_model.score(X_test, y_test)
            cv_scores.append(score)
        return print(f"Cross-validation Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")


    _()
    return


if __name__ == "__main__":
    app.run()

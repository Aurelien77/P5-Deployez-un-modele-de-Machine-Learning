import time
import pandas as pd
from IPython.display import display, HTML

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


def obtenir_grilles_cibles():
    return {
        "LogisticRegression_None": {
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__max_iter": [2000],
        },
        "LogisticRegression_L1": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["liblinear", "saga"],
            "classifier__max_iter": [2000],
        },
        "LogisticRegression_L2": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__solver": ["lbfgs", "liblinear", "saga"],
            "classifier__max_iter": [2000],
        },
        "ElasticNet_LogReg": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__l1_ratio": [0.1, 0.5, 0.9],
            "classifier__max_iter": [2000],
        },
        "Ridge": {
            "classifier__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "DecisionTree": {
            "classifier__max_depth": [None, 4, 8, 12, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__criterion": ["gini", "entropy"],
        },
        "RandomForest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 8, 16],
            "classifier__min_samples_leaf": [1, 2],
        },
        "GradientBoosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [2, 3],
        },
        "AdaBoost": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.05, 0.1, 0.5],
        },
        "XGBoost": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        },
        "SVR_Linear": {
            "classifier__C": [0.1, 1, 10],
            "classifier__loss": ["squared_hinge"],
            "classifier__dual": [False],
            "classifier__max_iter": [20000],
        },
        "SVC_RBF": {
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale", 0.01],
        },
        "SVC_Poly": {
            "classifier__C": [0.1, 1, 10],
            "classifier__degree": [2, 3],
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 11],
            "classifier__weights": ["uniform", "distance"],
        },
    }

def optimiser_modeles_cibles(
    models, preprocessor, cv, X_train, y_train, X_test, y_test,
    param_grids=None, scoring_grid='roc_auc', tri_par='F1_Score', verbose=True,
):
    if param_grids is None:
        param_grids = obtenir_grilles_cibles()

    tuned_results = []
    best_pipelines = {}

    for name, clf in models.items():
        if verbose:
            print(f"⚙️ Optimisation approfondie en cours pour : {name}...")

        pipeline = Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', clone(clf)),
        ])

        grid = param_grids.get(name, {})
        start_time = time.time()

        if grid:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                cv=cv,
                scoring=scoring_grid,
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            best_cv_score = grid_search.best_score_
            if verbose:
                print(f"    -> Meilleurs paramètres : {grid_search.best_params_}")
        else:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring_grid)
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_cv_score = cv_scores.mean()

        duration = time.time() - start_time
        best_pipelines[name] = best_pipeline

        if name in ["SVR_Linear", "Ridge"] or not hasattr(best_pipeline, "predict_proba"):
            try:
                decisions = best_pipeline.decision_function(X_test)
                y_pred_proba = 1 / (1 + np.exp(-decisions))
            except Exception:
                y_pred_proba = [0.5] * len(y_test)
        else:
            y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

        y_pred = best_pipeline.predict(X_test)

        tuned_results.append({
            "Model": f"{name} (Tuned Deep)",
            "ROC_AUC_cv": best_cv_score,
            "ROC_AUC_test": roc_auc_score(y_test, y_pred_proba),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1_Score": f1_score(y_test, y_pred, zero_division=0),
            "Duration_s": duration,
        })

        if verbose:
            print(f"    -> ROC-AUC Test : {tuned_results[-1]['ROC_AUC_test']:.4f}\n")

    df_tuned_res = pd.DataFrame(tuned_results).set_index("Model")
    
    # Sécurité pour éviter un plantage si la colonne de tri demandée n'existe pas
    sort_column = tri_par if tri_par in df_tuned_res.columns else "F1_Score"
    df_tuned_res = df_tuned_res.sort_values(sort_column, ascending=False)
    
    return df_tuned_res, best_pipelines


def afficher_resultats_tuning_cibles(df_tuned_res, titre="📊 Comparatif des modèles ciblés après GridSearch approfondi"):
    display(HTML(f"<h3>{titre}</h3>"))
    display(
        df_tuned_res.style
        .format(precision=3)
        .background_gradient(subset=["ROC_AUC_cv", "ROC_AUC_test", "F1_Scale" if "F1_Scale" in df_tuned_res.columns else "F1_Score", "Recall"], cmap="Blues")
    )
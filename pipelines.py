from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline as ImbPipeline

class ValueMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict):
        self.mapping_dict = mapping_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mapping_dict.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(mapping)
        return X_copy
    
class AbsoluteValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                # X_copy[col] = X_copy[col].abs()
                X_copy[col] = -X_copy[col]  # more robust
        return X_copy
    
    
    
    
def get_preprocessor(onehot_cols, mapping, negative_cols, model_type='nn'):
    if model_type == 'nn':
        # OneHot
        col_trans = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(handle_unknown='error', sparse_output=False), onehot_cols)
        ], remainder='passthrough')

    elif model_type == 'rf':
        # Label Encoding
        col_trans = ColumnTransformer(transformers=[
            ('labelenc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), onehot_cols)
        ], remainder='passthrough')

    else:
        raise ValueError("model_type should be 'nn' ou 'rf'")

    preprocessor = Pipeline(steps=[
        ('mapper', ValueMapper(mapping)),
        ('positive_transform', AbsoluteValueTransformer(columns=negative_cols)),
        ('column_trans', col_trans)
    ])
    return preprocessor
#_______________________________________________________________
#_______________________________________________________________
#_______________________________________________________________

# def get_randomforest_pipeline(onehot_cols, mapping, negative_cols, imbalance_method='none', hyperparameters=None):
#     default_rf_params = {
#         'n_estimators': 100,
#         'max_depth': 10,
#         'min_samples_split': 5,
#         'min_samples_leaf': 2,
#         'random_state': 0
#     }
    
#     if hyperparameters:
#         default_rf_params.update(hyperparameters)
    
#     preprocessor = get_preprocessor(onehot_cols, mapping, negative_cols, model_type='rf')
    
#     if imbalance_method == 'balanced':
#         # Use class_weight='balanced' to handle imbalance
#         rf_classifier = RandomForestClassifier(
#             class_weight='balanced',
#             **default_rf_params
#         )
#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('classifier', rf_classifier)
#         ])
        
#     elif imbalance_method == 'undersampling':
#         rf_classifier = RandomForestClassifier(**default_rf_params)
#         # Preprocessor needs to be applied before undersampling
#         preprocessed_pipeline = Pipeline([('preprocessor', preprocessor)])
#         pipeline = ImbPipeline([
#             ('preprocessed', preprocessed_pipeline),
#             ('undersampler', RandomUnderSampler(random_state=0)),
#             ('classifier', rf_classifier)
#         ])
#     elif imbalance_method == 'smote':
#         rf_classifier = RandomForestClassifier(**default_rf_params)
#         pipeline = ImbPipeline([
#             ('preprocessor', preprocessor),
#             ('smote', SMOTE(random_state=0)),
#             ('classifier', rf_classifier)
#         ])
        
#     else:  
#         # Classic RF without any imbalance handling
#         rf_classifier = RandomForestClassifier(**default_rf_params)
#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('classifier', rf_classifier)
#         ])
    
#     return pipeline





# def evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test, method_name):
    
#     pipeline.fit(X_train, y_train)
    
#     y_pred = pipeline.predict(X_test)
#     y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
#     results = {}
#     results['classification_report'] = classification_report(y_test, y_pred)
#     results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
#     results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
#     precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
#     results['pr_auc'] = auc(recall, precision)
    
#     cv_scores = cross_val_score(
#         pipeline, X_train, y_train, 
#         cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1),
#         scoring='roc_auc'
#     )
#     results['cv_mean'] = cv_scores.mean()
#     results['cv_std'] = cv_scores.std()
    
#     print(f"\n{'='*50}")
#     print(f"RESULTS - {method_name}")
#     print(f"{'='*50}")
#     print(f"ROC AUC: {results['roc_auc']:.4f}")
#     print(f"Precision-Recall AUC: {results['pr_auc']:.4f}")
#     print(f"CV Mean: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
#     print("\nClassification Report:")
#     print(results['classification_report'])
#     print("\nConfusion Matrix:")
#     print(results['confusion_matrix'])
    
#     return results, pipeline
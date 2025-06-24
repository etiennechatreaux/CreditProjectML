import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_distributions(df):
    print("Numerical cols:")
    plot_numerical_distributions(df)
    print("Categorical cols:")
    plot_categorical_distributions(df)
    
    
def plot_numerical_distributions(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3  # 3 plots per row, rounded up
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten array for easier indexing
    
    for idx, col in enumerate(numerical_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df):
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3  # 3 plots per row, rounded up
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten array for easier indexing
    
    for idx, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    for idx in range(len(categorical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    
def plot_distributions_by_target(df, columns_to_plot):
    for col in columns_to_plot:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")
    
    n_cols = len(columns_to_plot)
    n_rows = (n_cols + 2) // 3  # 3 plots per row, rounded up
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten array for easier indexing
    
    for idx, col in enumerate(columns_to_plot):
        sns.histplot(data=df[df['TARGET']==0], x=col, color='blue', alpha=0.5, 
                    label='TARGET=0', ax=axes[idx], kde=True)
        sns.histplot(data=df[df['TARGET']==1], x=col, color='red', alpha=0.5,
                    label='TARGET=1', ax=axes[idx], kde=True)
        
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].legend()
    
    for idx in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    
def plot_correlation_matrix(df, only_target=True):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if only_target:
        corr = df[numerical_cols].corr()['TARGET'].sort_values(ascending=False)
        plt.figure(figsize=(15, 6))
        corr.plot(kind='bar')
        plt.title('Correlations with TARGET')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(15, 12))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation matrix')
        plt.show()
    return corr
    

def get_top_correlations(corr_array, n_top=10):
    if not isinstance(corr_array, pd.DataFrame):
        corr_array = pd.DataFrame(corr_array)
    
    mask = ~np.eye(corr_array.shape[0], dtype=bool)
    
    abs_corr = np.abs(corr_array.values)
    abs_corr[~mask] = 0
    
    nan_mask = ~np.isnan(abs_corr)
    abs_corr[~nan_mask] = 0
    
    i, j = np.unravel_index(np.argsort(abs_corr, axis=None)[-n_top:], abs_corr.shape)
    
    top_corrs = []
    for idx in range(len(i)):
        feat1 = corr_array.index[i[idx]]
        feat2 = corr_array.columns[j[idx]]
        corr_value = corr_array.iloc[i[idx], j[idx]]
        if not np.isnan(corr_value):
            top_corrs.append((feat1, feat2, corr_value))
    
    top_corrs = sorted(top_corrs, key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\nTop {n_top} correlations:")
    for feat1, feat2, corr in top_corrs:
        print(f"{feat1} -- {feat2}: {corr:.3f}")
    return top_corrs
    
# ________________________________________________________________

def get_transformed_feature_names(preprocessor, raw_dataset, onehot_cols):
    
    onehot_feature_names = preprocessor.named_steps['column_trans'].named_transformers_['onehot'].get_feature_names_out(onehot_cols)
    original_feature_names = [col for col in raw_dataset.columns if col not in onehot_cols]
    return list(onehot_feature_names) + original_feature_names
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description='PCA visualization of customer churn')
    parser.add_argument('--data', default=str(ROOT / 'data' / 'telecom_churn.csv'))
    args = parser.parse_args()
    output = ROOT / 'output'
    output.mkdir(exist_ok=True)
    df = pd.read_csv(args.data)
    target = next((c for c in df.columns if c.lower() in ('churn', 'churn_label', 'exited')), None)
    if target is None:
        raise ValueError('No churn label found. Expected Churn, churn_label, or Exited column.')
    y = df[target].astype(str).fillna('Unknown')
    X = df.drop(columns=[target])
    X = X.drop(columns=[c for c in X.columns if c.lower() in ('customerid', 'customer_id', 'id', 'name', 'surname')], errors='ignore')
    X = X.replace(r'^\s*$', np.nan, regex=True)
    for col in X.columns:
        if X[col].dtype == 'object':
            converted = pd.to_numeric(X[col], errors='coerce')
            if converted.notna().sum() >= 0.9 * X[col].notna().sum() and converted.notna().sum() > 0:
                X[col] = converted
    for col in X.select_dtypes(include='number'):
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(exclude='number'):
        X[col] = X[col].fillna('Unknown')
    X = pd.get_dummies(X, drop_first=False, dtype=float)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] < 2:
        raise ValueError('At least two nonconstant encoded features are required.')
    scaled = StandardScaler().fit_transform(X)
    pca95 = PCA(n_components=0.95, svd_solver='full').fit(scaled)
    pca2 = PCA(n_components=2).fit(scaled)
    coords = pca2.transform(scaled)
    result = pd.DataFrame({'PC1': coords[:, 0], 'PC2': coords[:, 1], 'Churn': y})
    result.to_csv(output / 'pca_2d_coordinates.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, group in result.groupby('Churn'):
        ax.scatter(group.PC1, group.PC2, alpha=0.6, s=22, label=str(label))
    ax.set(xlabel=f'PC1 ({pca2.explained_variance_ratio_[0]:.1%} variance)', ylabel=f'PC2 ({pca2.explained_variance_ratio_[1]:.1%} variance)', title='Customer Churn: PCA 2D Visualization')
    ax.legend(title='Churn')
    fig.tight_layout()
    fig.savefig(output / 'pca_churn_scatter.png', dpi=180)
    plt.close(fig)
    cum = np.cumsum(PCA().fit(scaled).explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum) + 1), cum, marker='o', markersize=3)
    ax.axhline(0.95, linestyle='--', label='95% variance')
    ax.set(xlabel='Number of principal components', ylabel='Cumulative explained variance', title='PCA Explained Variance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'pca_explained_variance.png', dpi=180)
    plt.close(fig)
    summary = (f'Rows: {len(df)}\nEncoded features: {X.shape[1]}\nComponents to retain >=95% variance: {pca95.n_components_}\n'
               f'Variance retained by 2 components: {pca2.explained_variance_ratio_.sum():.2%}\n'
               'Interpretation: Inspect pca_churn_scatter.png. Overlapping churn groups indicate weak visual separation; distinct clusters indicate stronger separation. PCA is unsupervised and 2D may lose information.\n')
    (output / 'conclusion.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print('Saved outputs to:', output)

if __name__ == '__main__':
    main()

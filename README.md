# PCA Customer Churn Visualization

## Aim
Use Principal Component Analysis (PCA) to create a 2D representation of customer churn data and visualize churn labels.

## Steps
1. Read a CSV dataset and separate the `Churn` label.
2. Impute missing values, one-hot encode categorical features, and standardize the features.
3. Fit PCA retaining at least 95% variance and report the number of components required.
4. Fit a separate two-component PCA for the scatter plot. **Two components are not guaranteed to retain 95% variance.**
5. Save the scatter plot, cumulative explained variance plot, coordinates, and conclusion.

## Run in VS Code terminal
```powershell
python -m pip install -r requirements.txt
python pca_churn.py
```

Or specify your own churn CSV:
```powershell
python pca_churn.py --data "data/your_file.csv"
```

## Files
- `data/telecom_churn.csv`: synthetic example data for demonstration, **not real customer records**.
- `pca_churn.py`: full preprocessing and PCA analysis.
- `output/pca_churn_scatter.png`: churn-colored PCA plot.
- `output/pca_explained_variance.png`: cumulative explained variance plot.
- `output/pca_2d_coordinates.csv`: PCA coordinates and churn labels.
- `output/conclusion.txt`: summary and interpretation guidance.

## Conclusion
PCA reduces many standardized churn features to a smaller number of components. A 2D visualization helps inspect potential group separation, but PCA does not use the churn labels to find components and visual overlap does not by itself determine model accuracy.

# Preprocessing notes

- Passing is defined here as `G3 >= 10`; this is the project's chosen threshold.
- Removed 0 exact duplicate source rows. Missing-value rows removed after deduplication: 0.
- Excluded G1, G2, and G3 from predictors because they are grades from the same course and would allow the model to copy information directly related to the target.
- Checked 16 numeric columns using boxplots and the 1.5 x IQR rule. Potential outlier count across columns (one observation can be counted in several columns): 212; retained all observations because unusual values are not automatically errors. Counts are in `outputs/iqr_outlier_counts.csv`.
- Stratified split with seed 42: 276 train, 59 validation, 60 test observations.
- One ColumnTransformer uses one-hot encoding for categories and standardization for numeric features; it is fitted on the training partition only.

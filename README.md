# What Drives the Price of a Car?

This project uses the CRISP-DM framework to analyze 426,880 used-vehicle listings and identify the characteristics most strongly associated with asking price. The intended audience is a used-car dealership seeking practical guidance for inventory selection and pricing.

## Notebook

[Open the complete analysis notebook](prompt_II.ipynb)

The notebook contains the full data audit, cleaning decisions, exploratory visualizations, preprocessing pipelines, cross-validation, hyperparameter grid search, coefficient interpretation, held-out evaluation, and final business recommendations.

## Key findings

- **Vehicle age was the strongest predictor:** newer vehicles generally had higher asking prices.
- **Mileage was the second-largest driver:** lower-odometer vehicles generally retained more value.
- **Drive type and vehicle type added important pricing information** beyond age and mileage.
- Manufacturer, fuel type, condition, and transmission also contributed useful predictive signal.
- A tuned decision-tree regression achieved a held-out **MAE of $3,926**, **RMSE of $6,293**, and **R² of 0.807**.
- Ridge coefficients supported the directional findings: vehicle age and mileage had large negative conditional associations with asking price, while several premium manufacturers, diesel fuel, new condition, and truck body type had positive associations.

## Recommendations for the dealership

1. Prioritize newer, lower-mileage inventory when acquisition cost still leaves an acceptable margin.
2. Set acquisition and pricing thresholds by vehicle segment and drivetrain instead of using a single rule for all inventory.
3. Use the tuned model as a pricing guardrail to flag potentially over- or underpriced vehicles for human review.
4. Record standardized condition grades and actual selling prices to improve future models.
5. Send rare, premium, or otherwise unusual vehicles to specialist appraisal rather than relying only on the mass-market model.

## Modeling approach

The analysis compares a median-price baseline, Ridge regression, and decision-tree regression using three-fold cross-validation. `GridSearchCV` tunes tree depth and minimum leaf size using mean absolute error (MAE) as the selection metric. MAE is the primary metric because it expresses the typical prediction error directly in dollars and is less sensitive to a few unusually large errors than squared-error metrics.

The selected model is evaluated once on an untouched test set. Permutation importance explains the tuned tree's predictions, while Ridge coefficients provide directional, conditional associations.

## Data preparation

The workflow:

- removes nonpositive and implausibly large prices;
- limits unrealistic model years and odometer values;
- removes duplicate listing IDs;
- excludes identifier and extremely incomplete fields from modeling;
- imputes missing numeric and categorical values inside the modeling pipeline; and
- uses one-hot encoding for categorical predictors.

The source dataset is located at `data/vehicles.csv`.

## Reproduce the analysis

1. Open `prompt_II.ipynb` in JupyterLab.
2. Confirm that `data/vehicles.csv` is present.
3. Select a Python environment containing pandas, NumPy, matplotlib, seaborn, and scikit-learn.
4. Restart the kernel and run all cells from top to bottom.

The notebook uses a fixed random seed for repeatable sampling, train/test splitting, cross-validation, and model fitting.

## Limitations

The target represents advertised asking price rather than confirmed sale price. The data do not include acquisition cost, reconditioning expense, days to sell, or dealership profit. The results therefore describe predictive associations rather than causal consumer preferences and should support—not replace—inspection and local market judgment.

## Project structure

```text
practical_application_II_starter (1)/
├── README.md
├── prompt_II.ipynb
├── data/
│   └── vehicles.csv
└── images/
```

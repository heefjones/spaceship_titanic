# Spaceship Titanic
This [Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview) tasked me to predict if a passenger was **"Transported"** to another dimension after the Spaceship Titanic collided with a spacetime anomaly.  

## Data
The dataset is a fictional version of the [Titanic](https://www.kaggle.com/competitions/titanic) competition.
- **Rows:** ~8700  
- **Columns:** 13 (including passenger identifier and target label)  

### Null Values
- **Age:** Filled nulls with the median due to a near-normal distribution with a slight right skew.  
- **Monetary Columns:** RoomService, FoodCourt, ShoppingMall, Spa — most values were 0 with occasional high spenders. Nulls were filled with 0.  
- **String/Object Columns:** Filled nulls with `"Unknown"` to create a separate category.

## Feature Engineering
- **Total Expense:** Sum of the 5 monetary columns.  
- **Polynomial & Interaction Terms:** Generated additional features, expanding the feature set to 41 total.

## Modeling
- **Model:** XGBoost Classifier  
- **Hyperparameter Tuning:** Bayesian Optimization with 110 iterations  
- **Results:**
    - 79.47% accuracy on a 20% unseen test set.
    - Final Kaggle submission achieved **80.12% accuracy**.

## Files
- 📊 submission.ipynb – EDA, feature engineering, model iteration, and final submission.
- 🛠️ helper.py – Custom functions for data processing and model training.
- 📈 models_df.csv – Model performance metrics.

## Repository Structure
```
/spaceship_titanic
├── submission.ipynb
├── helper.py
├── models_df.csv
└── README.md
```

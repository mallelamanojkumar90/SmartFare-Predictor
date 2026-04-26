import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Constants for synthetic data generation
ZONES = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
TIMES = ['morning', 'afternoon', 'evening', 'night']
WEATHER = ['sunny', 'rainy', 'cloudy']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def generate_synthetic_data(n_samples=5000):
    """
    Generates a synthetic dataset simulating NYC taxi patterns for dynamic pricing.
    """
    np.random.seed(42)

    data = {
        'demand': np.random.randint(10, 100, n_samples),
        'supply': np.random.randint(10, 100, n_samples),
        'pickup_zone': np.random.choice(ZONES, n_samples),
        'time_of_day': np.random.choice(TIMES, n_samples),
        'day_of_week': np.random.choice(DAYS, n_samples),
        'weather': np.random.choice(WEATHER, n_samples),
        'trip_distance': np.random.uniform(0.5, 20.0, n_samples)
    }

    df = pd.DataFrame(data)

    # Calculate target: price_multiplier
    # Base multiplier is 1.0. It increases with demand/supply ratio.
    # We add specific boosts for rain and peak hours.

    df['demand_supply_ratio'] = df['demand'] / df['supply']

    # Base multiplier logic
    multiplier = 1.0 + (df['demand_supply_ratio'] - 1.0) * 0.5

    # Weather boost
    weather_boost = df['weather'].map({'rainy': 0.3, 'cloudy': 0.1, 'sunny': 0.0})
    multiplier += weather_boost

    # Time of day boost (Peak hours)
    time_boost = df['time_of_day'].map({'morning': 0.2, 'evening': 0.3, 'afternoon': 0.0, 'night': 0.1})
    multiplier += time_boost

    # Day of week boost (Weekends)
    day_boost = df['day_of_week'].apply(lambda x: 0.2 if x in ['Saturday', 'Sunday'] else 0.0)
    multiplier += day_boost

    # Add some random noise
    multiplier += np.random.normal(0, 0.1, n_samples)

    # Clip multiplier to reasonable range [1.0, 4.0]
    df['price_multiplier'] = multiplier.clip(1.0, 4.0)

    return df

def train_model():
    # 1. Data Generation
    print("Generating synthetic data...")
    df = generate_synthetic_data()

    # Feature Engineering
    df['is_peak'] = df['time_of_day'].apply(lambda x: 1 if x in ['morning', 'evening'] else 0)

    X = df.drop(columns=['price_multiplier'])
    y = df['price_multiplier']

    # 2. Preprocessing Pipeline
    categorical_cols = ['pickup_zone', 'time_of_day', 'day_of_week', 'weather']
    numerical_cols = ['demand', 'supply', 'trip_distance', 'demand_supply_ratio', 'is_peak']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # 3. Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Hyperparameter Tuning
    print("Tuning hyperparameters...")
    param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__subsample': [0.8, 1.0]
    }

    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # 5. Evaluation
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}")

    # Feature Importance
    # We need to get the feature names after one-hot encoding
    cat_features = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_features = numerical_cols + list(cat_features)

    importances = best_model.named_steps['regressor'].feature_importances_
    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feat_imp.head(15).plot(kind='barh')
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('ride_pricing_app/model/feature_importance.png')
    print("Feature importance plot saved to ride_pricing_app/model/feature_importance.png")

    # 6. Export
    print("Saving model and preprocessing pipeline...")
    joblib.dump(best_model, 'ride_pricing_app/model/model.pkl')

    print("Training complete!")

if __name__ == "__main__":
    train_model()

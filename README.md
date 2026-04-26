# SmartFare Predictor - Ride Sharing Dynamic Pricing System

This project implements an end-to-end dynamic pricing system for ride-sharing services. It uses a machine learning model (XGBoost) to predict price multipliers based on supply, demand, weather, and time factors.

## Project Structure
```
ride_pricing_app/
├── data/              # Dataset storage (empty in this version as we generate synthetic data)
├── model/             # Saved model and preprocessing pipeline
│   ├── model.pkl      # Trained XGBoost pipeline
│   └── feature_importance.png
├── app.py            # Streamlit web application
├── train.py          # Model training and evaluation script
└── requirements.txt   # Project dependencies
```

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <<repositoryrepository-url>
   cd ride_pricing_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   Generate the synthetic dataset and train the model to create the `model.pkl` file:
   ```bash
   python train.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Deployment to Hugging Face Spaces

The application is deployed and available at: [SmartFare Predictor on Hugging Face Spaces](https://huggingface.co/spaces/mallelamanoj75/ride-pricing-app)

### How to deploy your own version:
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Upload the following files to the Space:
   - `app.py`
   - `requirements.txt`
   - `model/model.pkl`
   - `model/feature_importance.png`
4. The Space will automatically install dependencies and start the app.

## Model Details
- **Algorithm**: XGBoost Regressor.
- **Features**: Demand, Supply, Pickup Zone, Time of Day, Day of Week, Weather, Trip Distance.
- **Target**: Price Multiplier (1.0x to 4.0x).
- **Evaluation**: Optimized using `RandomizedSearchCV` for minimum MAE.


# House Price Prediction API

This project is a Machine Learning-powered House Price Prediction API built using Flask.It allows users to input housing features and get price predictions from multiple trained models.

## 📌 Project Structure
```
house_price_prediction/
├── README.md
├── api/
│ └── app.py
├── data/
│ ├── housing.csv
│ └── processed_data.csv
├── logs/
│ └── training.log
├── models/
├── scripts/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ └── train.py
├── venv/
├── house_price_prediction.ipynb
├── requirements.txt
└── report.pdf
```
## ⚙️ Setup Instructions

1. Clone repository:
```
 git clone https://github.com/CHekri2123/house-price-prediction
 cd house_price_prediction
```
2. Create a Virtual Environment (Optional but Recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install Dependencies:
```
pip install -r requirements.txt
```
4. Run the API:
```
cd api/
python app.py
```
The API will start at http://127.0.0.1:5001

## Using the API
Endpoint: `/predict`

- Method: `POST`
- Sample Request (`Using curl`)
```
curl -X POST "http://127.0.0.1:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "longitude": -122.23,
          "latitude": 37.88,
          "housing_median_age": 41.0,
          "rooms_per_household": 6.984127,
          "bedrooms_per_room": 0.146590,
          "log_median_income": 11.45,
          "log_population_per_household": 2.531
     }'
```
- Sample Response
```
{
  "predictions": {
    "Decision Tree": 497886.61,
    "Lasso Regression": 2383446.49,
    "Linear Regression": 2383446.49,
    "Random Forest": 452592.8,
    "Ridge Regression": 2250938.1
  }
}
```
## Model Training & Data Processing
 - **Data Preprocessing**: Missing values handled, categorical encoding, feature scaling.
- **Feature Engineering**: New features created for better model accuracy.
- **Models Trained**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest.
- **Evaluation Metrics**: RMSE, MAE, R²-score.

##  License
This project is open-source under the **MIT License**.



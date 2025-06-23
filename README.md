# 📈 Stock Predictor App

## Overview
The Stock Predictor App is an interactive Streamlit application that predicts the next day's closing price of a stock using a deep learning model. The app fetches the last 10 days of historical stock data, preprocesses it, and uses a trained Keras model to forecast the next closing price. It also provides a detailed explanation for the prediction and allows users to download a PDF report of the results.

---

## Features

- **Stock Data Fetching:** Retrieves the last 10 days of daily stock prices for a user-specified symbol using the Twelve Data API.
- **AI-Powered Prediction:** Uses a pre-trained TensorFlow Keras model to predict the next day's closing price.
- **Data Preprocessing:** Automatically scales and prepares features for the model, including date-based features.
- **Interactive Visualization:** Plots historical prices, the predicted price, and a ±3% confidence band for easy interpretation.
- **Reasoning and Explanation:** Generates a human-readable explanation for the prediction using agent-based reasoning.
- **PDF Report Generation:** Allows users to download a PDF report containing the prediction and its explanation.
- **Astra DB Integration:** Stores predictions and explanations in a Cassandra-based Astra DB vector store for future reference.

---

## Pipeline

The machine learning pipeline (data collection, preprocessing, model training, and evaluation) for this app is managed separately and is available at:  
👉 [Stock Sage ML Pipeline](https://github.com/shashwat051102/Stock_Sage_ML_pipeline)

---

## Installation

### Prerequisites

- Python 3.10+
- Streamlit
- TensorFlow
- Twelve Data API Key
- Astra DB credentials

### Steps

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-repo/Stock_Predictor_App.git](https://github.com/shashwat051102/Stock_sage_app.git)
   cd Stock_Predictor_App
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```env
     STOCK_API=<your_twelve_data_api_key>
     ASTRA_DB_APPLICATION_TOKEN=<your_astra_db_application_token>
     ASTRA_DB_ID=<your_astra_db_id>
     ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Enter a valid stock symbol (e.g., `AAPL`, `GOOG`, `TSLA`) in the input field.
2. Click **Fetch & Predict** to retrieve historical data and predict the next day's closing price.
3. View the prediction results, explanation, and interactive chart.
4. Download the PDF report for detailed insights.

---

## How It Works

1. **User Input:**  
   The user enters a stock symbol and clicks "Fetch & Predict".

2. **Data Fetching:**  
   The app requests the last 10 days of daily price data from the Twelve Data API.

3. **Data Preprocessing:**  
   - Converts price columns to float.
   - Extracts year, month, and day as features.
   - Scales features and target using MinMaxScaler.

4. **Prediction:**  
   - Loads the pre-trained Keras model.
   - Predicts the next day's closing price using the scaled features.
   - Inverse transforms the prediction to get the actual price.

5. **Visualization:**  
   - Plots the last 10 days of closing prices.
   - Shows the predicted price for the next day.
   - Adds a ±3% confidence band for context.

6. **Explanation:**  
   - Uses agent-based reasoning to generate a human-readable explanation for the prediction.
   - Stores the prediction and explanation in Astra DB.

7. **PDF Report:**  
   - Allows the user to download a PDF report containing the prediction and explanation.

---

## File Structure

```
.env
.gitignore
app.py
requirements.txt
Agents/
    Reason_agent.py
model/
    best_model.keras
    feature_scaler.pkl
    target_scaler.pkl
Task/
    Reason_task.py
utils/
    Astra_DB_search.py
    Email_send.py
    pdf_download.py
```

### Key Files

- **`app.py`**: Main Streamlit application file.
- **`model/best_model.keras`**: Pre-trained Keras model for stock price prediction.
- **`utils/Astra_DB_search.py`**: Handles Astra DB vector store operations.
- **`utils/pdf_download.py`**: Generates and downloads PDF reports.

---

## Technologies Used

- **Frontend:** Streamlit
- **Backend:** Python
- **Machine Learning:** TensorFlow, Keras, scikit-learn
- **Database:** Astra DB (Cassandra)
- **Visualization:** Matplotlib

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Twelve Data API](https://twelvedata.com/)
- [Astra DB](https://www.datastax.com/products/datastax-astra)
- [Stock Sage ML Pipeline](https://github.com/shashwat051102/Stock_Sage_ML_pipeline)

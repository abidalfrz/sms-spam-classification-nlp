# 📩 SMS Spam Detection

The repository contains a Machine Learning-based web application that detects whether an SMS message is spam or not, specifically tailored for the Indonesian language. Built using Streamlit, the application integrates a trained LSTM (Long Short-Term Memory) model to provide accurate classifications and confidence scores.

---

## 🧠 Features

- Real-time Prediction: Classify single SMS messages instantly.
- Batch Processing: Upload CSV files to predict multiple messages at once.
- Interactive Visualizations:
  - Pie charts for confidence scores.
  - Bar charts for the distribution of Spam vs. Ham in uploaded files.
- Downloadable Reports: Export prediction results directly to CSV format.

---

## 🛠️ Tech Stack

### Frontend:

- **Language**: Python
- **Framework**: Streamlit

### Data Science & ML:

- **Data Handling**: Pandas
- **Numerical Computing**: NumPy
- **Data Visualization**: Plotly, Wordcloud
- **Text Processing**: NLTK, Sastrawi, Gensim
- **Machine Learning Algorithms**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning Frameworks**: TensorFlow / Keras
- **Hyperparameter Tuning**: Keras-Tuner

---

## 📁 Project Structure

```
sms-spam-classification-nlp/
├── app/
│   └── app.py                  # Main Streamlit application
│
├── assets/                     # Trained ML models and tokenizers
│   ├── best_lstm_model.h5      # The main trained LSTM model
│   ├── best_lstm_tuned_model.h5
│   ├── tokenizer.pickle        # Text tokenizer
│   └── word2vec.model          # Word embedding model
│
├── data/
│   └── sms_spam_indo.csv       # Raw dataset source
│
├── notebooks/                  # Data Science workspace
│   └── main.ipynb              # Model training and experimentation
│
├── utils/                      # Helper modules
│   ├── models.py               # Function to load models
│   └── preprocessing.py        # Text cleaning functions
│
├── .gitignore
├── requirements.txt            # Python dependencies list
└── README.md
```

---

## 🔁 Machine Learning Workflow

1. **Data Collection**: The dataset "sms_spam_indo.csv" contains labeled SMS messages in Indonesian.
2. **Exploratory Data Analysis (EDA)**: Initial analysis to understand data distribution and characteristics.
3. **Data Preprocessing**: Handled in `preprocessing.py` to clean and transform text data for modeling.
4. **Model Training** : Experiments with different algorithms in `main.ipynb`, leading to the selection of a LSTM model.
5. **Best Model Tuning**: Used keras-tuner for hyperparameter optimization to enhance model performance.
6. **Model Evaluation**: Performance metrics using AUC are calculated to ensure model reliability.
7. **Model Deployment**: The trained model is integrated into the Streamlit app for user interaction.

---

## 📂 Dataset & Credits

The dataset used in this project was sourced from Kaggle.  
You can access the original dataset and description through the link below:

🔗[SMS Spam Dataset](https://www.kaggle.com/datasets/gevabriel/indonesian-sms-spam)

We would like to acknowledge and thanks to the dataset creator for making this resource publicly available for research and educational use.

---

## 🚀 How to Run

### 1. Clone the Repository:

Open your terminal and run the following commands:

```bash
git clone https://github.com/abidalfrz/sms-spam-classification-nlp.git
cd sms-spam-classification-nlp
```

### 2. Create a Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate.bat     # On Windows
```

### 3. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application:

```bash
streamlit run app/app.py

# The webapp will be accessible at http://localhost:8501
```

### 5. Access the Application
Open your web browser and navigate to `http://localhost:8501` to interact with the SMS Spam Detection App.

1. Enter a single SMS message or upload a CSV file containing multiple messages.
2. Click the "Predict" button to see the classification results along with confidence scores and visualizations.

---






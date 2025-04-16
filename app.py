from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import io, base64
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import datetime

#import nltk
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

company_to_ticker = {
    # US Stocks
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "facebook": "META",
    "twitter": "TWTR",
    "netflix": "NFLX",
    "walmart": "WMT",
    "intel": "INTC",
    "coca-cola": "KO",
    "nike": "NKE",
    "disney": "DIS",
    "starbucks": "SBUX",
    "pepsi": "PEP",
    "boeing": "BA",
    "nvidia": "NVDA",
    "spotify": "SPOT",
    "salesforce": "CRM",
    
    # Indian Stocks (NSE)
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "hdfc": "HDFC.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "axis bank": "AXISBANK.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "sun pharma": "SUNPHARMA.NS",
    "l&t": "LT.NS",
    "maruti": "MARUTI.NS",
    "hindalco": "HINDALCO.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "nestle india": "NESTLEIND.NS",
    "tata motors": "TATAMOTORS.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "mahilakshmi": "MAHINDCIE.NS",
    "shree cement": "SHREECEM.NS",
    
    # UK Stocks
    "vodafone": "VOD.L",
    "tesco": "TSCO.L",
    "barclays": "BARC.L",
    "unilever": "ULVR.L",
    "bp": "BP.L",
    "glaxo smithkline": "GSK.L",
    
    # European Stocks
    "nestle": "NESN.SW",  # Switzerland
    "siemens": "SIE.DE",  # Germany
    "l'oréal": "OR.PA",  # France
    "sanofi": "SAN.PA",  # France
    "sap": "SAP.DE",  # Germany
    "asml": "ASML.AS",  # Netherlands
    
    # Chinese Stocks
    "alibaba": "BABA",
    "tencent": "0700.HK",
    "baidu": "BIDU",
    "jd.com": "JD",
    "meituan": "3690.HK",
    "pinduoduo": "PDD",
    
    # Other global stocks
    "sony": "6758.T",  # Japan
    "toshiba": "6502.T",  # Japan
    "canon": "7751.T",  # Japan
    "lg electronics": "066570.KS",  # South Korea
    "samsung electronics": "005930.KS",  # South Korea
    "bayer": "BAYN.DE",  # Germany
    "abbvie": "ABBV",  # USA
    "ford": "F",  # USA
    "general electric": "GE",  # USA
    "abbott laboratories": "ABT",  # USA
    "general motors": "GM",  # USA
    "3m": "MMM"  # USA
}


app = Flask(__name__)

# Load models
lstm_model = load_model('models/stock_prediction_lstm.h5')
bilstm_model = load_model('models/stock_prediction_Bilstm.h5')

def get_model(name):
    if name == 'LSTM':
        return lstm_model
    elif name == 'BiLSTM':
        return bilstm_model
    return None

def get_prediction_plot(actual, predicted_price):
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    import base64

    actual = np.array(actual).flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(actual, label="Actual", color="blue")

    # Plot the predicted point
    if len(actual) > 0:
        last_index = len(actual) - 1
        plt.plot(last_index + 1, predicted_price, 'o', color='orange', label='Predicted')
        plt.plot([last_index, last_index + 1], [actual[-1], predicted_price], linestyle='--', color='orange')

    plt.title("Actual vs Predicted Stock Price")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64





def get_history_plot(df):
    plt.figure(figsize=(6, 3))
    plt.plot(df['Close'], label='Close Price')
    plt.title("Historical Close Price")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64


def get_ticker_from_query(query):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word not in stop_words]

    for word in filtered_words:
        if word in company_to_ticker:
            return company_to_ticker[word]
    return None

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "ticker": ticker,
            "name": info.get('shortName', 'N/A'),
            "current_price": info.get('currentPrice', 'N/A'),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A')
        }
    except Exception as e:
        return {"error": f"Error fetching data for {ticker}: {str(e)}"}
    
'''@app.route('/query', methods=['GET', 'POST'])
def query():
    result = None
    if request.method == 'POST':
        user_query = request.form['query']
        ticker = get_ticker_from_query(user_query)
        if ticker:
            result = get_stock_info(ticker)
        else:
            result = {"error": "Could not find a valid company in your query."}
    return render_template('query.html', result=result)'''


@app.route('/', methods=['GET', 'POST'])
def index():
    ticker = None
    predicted_price = None
    error = None
    graph_img = None
    history_img = None

    if request.method == 'POST':
        ticker = request.form['ticker'].strip().upper()
        model_name = request.form['model']
        model = get_model(model_name)

        try:
            df = yf.download(ticker, start="2012-01-01", end=datetime.date.today())
            if df.empty or 'Close' not in df.columns:
                raise ValueError("No data found")

            data = df[['Close']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Use last 60 days to predict next day
            input_seq = scaled_data[-100:]
            X_input = np.reshape(input_seq, (1, 100, 1))
            prediction = model.predict(X_input)
            predicted_price = scaler.inverse_transform(prediction)[0][0]

            # Generate graphs
            actual = data[-60:]
            #predicted_series = np.vstack((actual[1:], prediction))
            graph_img = get_prediction_plot(actual, predicted_price)
            history_img = get_history_plot(df)

        except Exception as e:
            error = f"Error during prediction: {str(e)}"

    return render_template('index.html', predicted_price=predicted_price,
                           error=error, graph_img=graph_img, history_img=history_img, ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True)

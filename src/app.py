import os
import logging
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Dynamically target and read the standalone tabular dataset file
CSV_SHOCKS_PATH = os.path.join("data", "historical_shocks.csv")

def load_historical_shocks():
    try:
        if os.path.exists(CSV_SHOCKS_PATH):
            df = pd.read_csv(CSV_SHOCKS_PATH)
            # Convert tabular dataframe back into dictionary records for JSON rendering
            return df.to_dict(orient="records")
        else:
            logging.warning(f"Shocks dataset file missing at {CSV_SHOCKS_PATH}. Using fallback dataset.")
            return []
    except Exception as e:
        logging.error(f"Failed to ingest tabular shocks file: {str(e)}")
        return []

@app.route('/api/v1/shocks', methods=['GET'])
def get_shocks():
    shocks_data = load_historical_shocks()
    return jsonify({
        "status": "success",
        "total_records": len(shocks_data),
        "data": shocks_data
    })

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "tabular_data_loaded": os.path.exists(CSV_SHOCKS_PATH)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
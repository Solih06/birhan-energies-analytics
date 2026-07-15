import os
import json
import logging
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for frontend components

RAW_PRICES_PATH = os.path.join("data", "raw", "BrentOilPrices.csv")
CSV_SHOCKS_PATH = os.path.join("data", "historical_shocks.csv")
BAYESIAN_OUTPUT_PATH = os.path.join("data", "bayesian_outputs.json")

# Core Data Access Utilities
def read_tabular_shocks():
    if os.path.exists(CSV_SHOCKS_PATH):
        try:
            return pd.read_csv(CSV_SHOCKS_PATH).to_dict(orient="records")
        except Exception as e:
            logging.error(f"Failed parsing shocks asset: {str(e)}")
    return []

def read_bayesian_json():
    if os.path.exists(BAYESIAN_OUTPUT_PATH):
        try:
            with open(BAYESIAN_OUTPUT_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed reading pre-calculated metrics: {str(e)}")
    return {}

# FLASK SERVICE ROUTE API DEFINITIONS

@app.route('/api/v1/prices', methods=['GET'])
def get_historical_time_series():
    """Streams cleaned price records with optional chronological query filtering."""
    if not os.path.exists(RAW_PRICES_PATH):
        return jsonify({"status": "error", "message": "Source database prices unavailable."}), 404
        
    try:
        df = pd.read_csv(RAW_PRICES_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate log returns array for dashboard plotting
        df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
        df = df.fillna(0)
        
        # Task 3 Requirement: Handle dynamic dashboard time slice filtering params
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
            
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        records = df.to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "count": len(records),
            "data": records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/v1/shocks', methods=['GET'])
def get_decoupled_shock_events():
    """Exposes decoupled CSV event vectors to overlay on interactive chart timelines."""
    events = read_tabular_shocks()
    return jsonify({
        "status": "success",
        "total_events": len(events),
        "data": events
    })

@app.route('/api/v1/change_points', methods=['GET'])
def get_bayesian_metrics():
    """Serves concrete parameters, HPDI brackets, and R-hat convergence values."""
    data_payload = read_bayesian_json()
    if not data_payload:
        return jsonify({
            "status": "pending",
            "message": "Bayesian MCMC estimation has not been executed yet."
        }), 202
        
    return jsonify({
        "status": "success",
        "data": data_payload
    })

@app.route('/api/v1/health', methods=['GET'])
def api_health_check():
    """Verifies infrastructure read states across all analytical subsystems."""
    return jsonify({
        "status": "operational",
        "dependencies": {
            "raw_prices_exist": os.path.exists(RAW_PRICES_PATH),
            "decoupled_events_exist": os.path.exists(CSV_SHOCKS_PATH),
            "bayesian_payload_computed": os.path.exists(BAYESIAN_OUTPUT_PATH)
        }
    })

if __name__ == "__main__":
    logging.info("Spinning up production Flask application on port 5000...")
    app.run(host="127.0.0.1", port=5000, debug=True)
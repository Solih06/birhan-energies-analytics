import os
import json
import logging
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing for React interaction

CSV_SHOCKS_PATH = os.path.join("data", "historical_shocks.csv")
BAYESIAN_OUTPUT_PATH = os.path.join("data", "bayesian_outputs.json")
RAW_PRICES_PATH = os.path.join("data", "raw", "BrentOilPrices.csv")

# Dynamic File Loader Utilities
def load_tabular_shocks():
    if os.path.exists(CSV_SHOCKS_PATH):
        try:
            df = pd.read_csv(CSV_SHOCKS_PATH)
            return df.to_dict(orient="records")
        except Exception as e:
            logging.error(f"Error reading historical shocks file: {str(e)}")
    return []

def load_bayesian_metrics():
    if os.path.exists(BAYESIAN_OUTPUT_PATH):
        try:
            with open(BAYESIAN_OUTPUT_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error parsing Bayesian JSON output: {str(e)}")
    return {}

# PRODUCTION API SERVICE ROUTES

@app.route('/api/v1/prices', methods=['GET'])
def get_historical_prices():
    """Streams full time-series rows with optional chronological query filtering."""
    if not os.path.exists(RAW_PRICES_PATH):
        return jsonify({"status": "error", "message": "Source pricing file not found."}), 404
        
    try:
        df = pd.read_csv(RAW_PRICES_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Apply optional dashboard query filters
        start_param = request.args.get('start_date')
        end_param = request.args.get('end_date')
        
        if start_param:
            df = df[df['Date'] >= pd.to_datetime(start_param)]
        if end_param:
            df = df[df['Date'] <= pd.to_datetime(end_param)]
            
        # Format dates back into string types for JSON serialization
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        price_records = df.to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "count": len(price_records),
            "data": price_records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/v1/shocks', methods=['GET'])
def get_historical_shocks():
    """Serves the decoupled 11 shock events to populate interactive dashboard bands."""
    shocks = load_tabular_shocks()
    return jsonify({
        "status": "success",
        "total_events": len(shocks),
        "data": shocks
    })

@app.route('/api/v1/change_points', methods=['GET'])
def get_bayesian_change_points():
    """Exposes calculated MCMC distributions, parameter changes, and HPDIs."""
    metrics = load_bayesian_metrics()
    if not metrics:
        return jsonify({
            "status": "pending", 
            "message": "Bayesian inference calculations not yet run or exported."
        }), 202
        
    return jsonify({
        "status": "success",
        "data": metrics
    })

@app.route('/api/v1/health', methods=['GET'])
def system_health_check():
    """Validates the availability of required system assets and structural metrics."""
    return jsonify({
        "status": "healthy",
        "assets": {
            "raw_prices_available": os.path.exists(RAW_PRICES_PATH),
            "tabular_shocks_available": os.path.exists(CSV_SHOCKS_PATH),
            "bayesian_outputs_calculated": os.path.exists(BAYESIAN_OUTPUT_PATH)
        }
    })

if __name__ == "__main__":
    logging.info("Starting production Flask REST API backend configuration...")
    app.run(host="127.0.0.1", port=5000, debug=True)
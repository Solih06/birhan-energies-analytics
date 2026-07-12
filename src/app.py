import os
import logging
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from src.data_pipeline import load_and_clean_data, apply_log_transformations

# Configure clean API execution logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for seamless React app connections

# Global variables to hold data structures in memory
DATA_PATH = os.path.join("data", "raw", "BrentOilPrices.csv")
PROCESSED_DF = None

def initialize_backend_data():
    """Loads and caches data in memory on server startup."""
    global PROCESSED_DF
    try:
        if os.path.exists(DATA_PATH):
            logging.info("API Server initializing data structures...")
            df = load_and_clean_data(DATA_PATH)
            PROCESSED_DF = apply_log_transformations(df)
            logging.info("Data loaded successfully and cached for API endpoints.")
        else:
            logging.warning(f"Data file not found at {DATA_PATH}. Some API endpoints will return empty structures.")
    except Exception as e:
        logging.error(f"Failed to preload data during API startup: {str(e)}")

# Expanded matrix featuring 11 historical shocks affecting Brent Crude Oil pricing structures
HISTORICAL_SHOCKS = [
    {"date": "1990-08-02", "event": "Gulf War Outbreak", "category": "Geopolitical Conflict", "description": "Iraq invades Kuwait, leading to immediate crude supply disruptions and soaring price volatility."},
    {"date": "1997-11-27", "event": "OPEC Jakarta Expansion Mistake", "category": "OPEC Policy", "description": "OPEC raises production quotas right as the Asian Financial Crisis triggers steep demand destruction."},
    {"date": "2001-09-11", "event": "September 11 Attacks", "category": "Geopolitical Shock", "description": "Global aviation contractions prompt immediate drops in jet fuel demand and safe-haven market premiums."},
    {"date": "2003-03-20", "event": "Iraq War Invasion", "category": "Geopolitical Conflict", "description": "US-led military operations disrupt Middle Eastern supply logistics, creating long-term structural risk premiums."},
    {"date": "2008-07-11", "event": "Global Financial Crisis Peak", "category": "Macroeconomic Crisis", "description": "Brent hits an all-time record high near $147/bbl before entering an aggressive demand-driven collapse."},
    {"date": "2011-02-15", "event": "Arab Spring Outbreak & Libya Crisis", "category": "Geopolitical Conflict", "description": "Civil unrest across MENA regions blocks Libyan sweet crude output, triggering structural price spikes."},
    {"date": "2014-11-27", "event": "OPEC Market Share Price War", "category": "OPEC Policy", "description": "OPEC maintains production volumes despite rising US shale output, shifting the market into a multi-year structural oversupply regime."},
    {"date": "2018-11-04", "event": "US Iran Sanctions Re-imposition", "category": "Geopolitical Shock", "description": "Re-imposition of secondary energy trade sanctions curtails Iranian export volumes from international shipping routes."},
    {"date": "2020-03-06", "event": "OPEC+ Alliance Breakdown", "category": "OPEC Policy", "description": "Saudi Arabia and Russia fail to agree on volume cuts, initiating a temporary market-share price war."},
    {"date": "2020-03-11", "event": "COVID-19 Pandemic Declaration", "category": "Macroeconomic Shock", "description": "WHO declares global pandemic status; widespread lockdowns cause unprecedented transport fuel demand collapse."},
    {"date": "2022-02-24", "event": "Russia-Ukraine War Outbreak", "category": "Geopolitical Conflict", "description": "Sanctions on Russian energy output prompt massive structural trade reorganizations and multi-year supply premiums."}
]
@app.route('/api/v1/health', methods=['GET'])
def server_health():
    """Simple system validation health-check route."""
    return jsonify({
        "status": "healthy",
        "service": "Birhan Energies Price Analytics Core Engine",
        "data_loaded": PROCESSED_DF is not None
    }), 200

@app.route('/api/v1/shocks', methods=['GET'])
def get_geopolitical_shocks():
    """Returns the matrix of researched historical events for dashboard timelines."""
    return jsonify({
        "status": "success",
        "count": len(HISTORICAL_SHOCKS),
        "events": HISTORICAL_SHOCKS
    }), 200

@app.route('/api/v1/metrics/summary', methods=['GET'])
def get_price_summary_statistics():
    """Calculates overall statistical baselines for high-level dashboard cards."""
    if PROCESSED_DF is None:
        return jsonify({"status": "error", "message": "Data structures uninitialized"}), 500
    
    summary = {
        "total_trading_days": int(len(PROCESSED_DF)),
        "max_price": float(PROCESSED_DF['Price'].max()),
        "min_price": float(PROCESSED_DF['Price'].min()),
        "mean_price": float(PROCESSED_DF['Price'].mean()),
        "overall_volatility_sigma": float(PROCESSED_DF['Log_Return'].std())
    }
    return jsonify({"status": "success", "metrics": summary}), 200

@app.route('/api/v1/prices', methods=['GET'])
def get_price_series():
    """
    Exposes full time-series rows for pricing charts.
    Supports a optional '?limit=X' query parameter to prevent front-end rendering lag.
    """
    if PROCESSED_DF is None:
        return jsonify({"status": "error", "message": "Data structures uninitialized"}), 500
    
    limit = request.args.get('limit', default=None, type=int)
    
    # Create web-ready dictionary structures from the cached DataFrame
    target_df = PROCESSED_DF.tail(limit) if limit else PROCESSED_DF
    
    records = []
    for _, row in target_df.iterrows():
        records.append({
            "date": row['Date'].strftime('%Y-%m-%d'),
            "price": float(row['Price']),
            "log_return": float(row['Log_Return']) if not pd.isna(row['Log_Return']) else 0.0,
            "is_outlier": int(row['Is_Outlier'])
        })
        
    return jsonify({
        "status": "success",
        "returned_records": len(records),
        "series": records
    }), 200

# Initialize global cache state before starting server routines
initialize_backend_data()

if __name__ == '__main__':
    logging.info("Starting Flask application API backend server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
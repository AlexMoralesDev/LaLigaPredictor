import json
import os

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React to call this API

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get all predictions from history"""
    try:
        with open('predictions_history.json', 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        return jsonify(predictions)
    except FileNotFoundError:
        return jsonify({"error": "No predictions found"}), 405

if __name__ == '__main__':
    app.run(debug=True, port=5001)

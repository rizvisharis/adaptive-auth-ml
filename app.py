# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load saved models
behavior_model = joblib.load('behavior_model.pkl')
context_model = joblib.load('context_model.pkl')

@app.route('/api/auth-score', methods=['POST'])
def auth_score():
    try:
        data = request.json
        behavior_input = data['behavior_input']
        context_input = data['context_input']
        expected_user = data['expected_user']

        # Convert to DataFrames
        X_behavior = pd.DataFrame([behavior_input]).fillna(0).astype(float)
        X_context = pd.DataFrame([context_input]).fillna(0).astype(str)

        # Predict probabilities
        prob_behavior = behavior_model.predict_proba(X_behavior)[0]
        prob_context = context_model.predict_proba(X_context)[0]

        # Find indexes
        i_b = np.where(behavior_model.classes_ == expected_user)[0][0]
        i_c = np.where(context_model.classes_ == expected_user)[0][0]

        # Extract probabilities
        p_behavior = prob_behavior[i_b]
        p_context = prob_context[i_c]

        # Weighted average
        w_behavior = 0.6
        w_context = 0.4
        combined_prob = w_behavior * p_behavior + w_context * p_context

        # Decide next auth step
        if combined_prob >= 0.80:
            step = "simple_auth"
        elif combined_prob >= 0.50:
            step = "security_question"
        else:
            step = "otp_verification"

        return jsonify({
            'combined_probability': round(combined_prob, 4),
            'next_step': step
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)

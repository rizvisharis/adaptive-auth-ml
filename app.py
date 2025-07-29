from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved models
behavior_model = joblib.load('best_behavior_model.pkl')
context_model = joblib.load('best_context_model.pkl')

# Load label encoders
behavior_encoder = joblib.load('behavior_label_encoder.pkl')
context_encoder = joblib.load('context_label_encoder.pkl')

@app.route('/api/auth-score', methods=['POST'])
def auth_score():
    try:
        data = request.json
        behavior_input = data['behavior_input']
        context_input = data['context_input']
        expected_user = data['expected_user']

        # Convert input to DataFrame
        X_behavior = pd.DataFrame([behavior_input]).fillna(0).astype(float)
        X_context = pd.DataFrame([context_input]).fillna(0).astype(str)

        # Predict probabilities
        prob_behavior = behavior_model.predict_proba(X_behavior)[0]
        prob_context = context_model.predict_proba(X_context)[0]

        # Convert expected_user to class ID
        expected_behavior_id = behavior_encoder.transform([expected_user])[0]
        expected_context_id = context_encoder.transform([expected_user])[0]

        # Get index of the class in the model's output
        i_b = behavior_model.classes_.tolist().index(expected_behavior_id)
        i_c = context_model.classes_.tolist().index(expected_context_id)

        # Extract individual probabilities
        p_behavior = prob_behavior[i_b]
        p_context = prob_context[i_c]

        # Weighted combination
        w_behavior = 0.5
        w_context = 0.5
        combined_prob = w_behavior * p_behavior + w_context * p_context
        print(f"Risk Score: {combined_prob * 100:.2f}%")

        # Determine next auth step
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

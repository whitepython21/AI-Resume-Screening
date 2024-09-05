from flask import Flask, render_template, request, jsonify
import joblib
from utils.data_preprocessing import clean_text, vectorize_text

app = Flask(__name__)

# Load trained model
model = joblib.load('model/sentiment_model.pkl')

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    feedback = request.form['feedback']
    cleaned_feedback = clean_text(feedback)
    vectorized_feedback = vectorize_text([cleaned_feedback])
    
    prediction = model.predict(vectorized_feedback)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    app.run(debug=True)

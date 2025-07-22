from flask import Flask, request, render_template
import pickle

model = pickle.load(open('genre_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    tagline = request.form.get('tagline', '')
    combined_text = f"{description} {tagline}"
    features = vectorizer.transform([combined_text])
    prediction = model.predict(features)
    return render_template('index.html', prediction_text=f"🎬 Predicted Genre: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)

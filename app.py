from flask import Flask, render_template, request
import joblib
import numpy as np 

app = Flask(__name__, template_folder='template')

model = joblib.load('model_popularite_post.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    features = [
        float(request.form['page_total_likes']),
        int(request.form['type']),
        int(request.form['category']),
        int(request.form['post_month']),
        int(request.form['post_weekday']),
        int(request.form['post_hour']),
        int(request.form['paid']),
        float(request.form['lifetime_post_total_reach']),
        float(request.form['lifetime_post_total_impressions']),
        float(request.form['lifetime_engaged_users']),
        float(request.form['lifetime_post_consumers']),
        float(request.form['lifetime_post_consumptions']),
        float(request.form['lifetime_post_impressions_by_likers']),
        float(request.form['lifetime_post_reach_by_likers']),
        float(request.form['lifetime_likers_engagement'])
    ]

    # Prédire les likes
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Prédiction des mentions J\'aime : {int(prediction[0])}')

if __name__ == "__main__":
    app.run(debug=True)


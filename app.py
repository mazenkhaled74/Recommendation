from flask import Flask, request, jsonify
import pandas as pd
from CoachRecommender import CoachRecommender
import joblib

app = Flask(__name__)

# Coaches Recommendation model startup
recommender = CoachRecommender(model_path='coach_recommender_model1.pkl')

# Load your coaches dataset once at startup
all_coaches = pd.read_csv("coach_suitability.csv")[['coach_id', 'coach_name', 'coach_rating', 'coach_experiences']].drop_duplicates().reset_index(drop=True)



# Plans Recommendation model startup
rf_exercises = joblib.load('rf_exercises_model.pkl')
rf_diet = joblib.load('rf_diet_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')


@app.route('/recommend/coaches', methods=['POST'])
def recommendCoaches():
    data = request.json

    required_fields = ['age', 'height', 'weight', 'body_fat', 'body_muscle', 'goals']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing one or more required fields.'}), 400

    try:
        top_experiences = recommender.recommend_coaches(data, all_coaches)
        return jsonify({'recommended_experiences': top_experiences})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend/exercise-diet', methods=['POST'])
def recommendPlans():
    try:
        data = request.get_json()

        required_fields = [
            'Sex', 'Age', 'Height', 'Weight', 'BMI',
            'Hypertension', 'Diabetes'
        ]
        missing = [field for field in required_fields if field not in data]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing fields: {', '.join(missing)}"
            }), 400

        # Convert to model input format
        user_features = [
            int(data['Sex']),
            int(data['Age']),
            float(data['Height']),
            float(data['Weight']),
            float(data['BMI']),
            int(data['Hypertension']),
            int(data['Diabetes'])
        ]

        # Predict Exercise & Diet
        exercise_pred = rf_exercises.predict([user_features])[0]
        diet_pred = rf_diet.predict([user_features])[0]

        exercise_label = label_encoders['Exercises'].inverse_transform([exercise_pred])[0]
        diet_label = label_encoders['Diet'].inverse_transform([diet_pred])[0]

        return jsonify({
            "status": "success",
            "exercise": exercise_label,
            "diet": diet_label
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    print("Service is running")
    app.run(host="0.0.0.0", port=5000)

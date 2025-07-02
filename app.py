from flask import Flask, request, jsonify
import pandas as pd
from CoachRecommender import CoachRecommender

app = Flask(__name__)
recommender = CoachRecommender(model_path='coach_recommender_model.pkl')

# Load your coaches dataset once at startup
all_coaches = pd.read_csv("coach_suitability.csv")[['coach_id', 'coach_name', 'coach_rating', 'coach_experiences']].drop_duplicates().reset_index(drop=True)

@app.route('/recommend/coaches', methods=['POST'])
def recommend():
    data = request.json

    required_fields = ['age', 'height', 'weight', 'body_fat', 'body_muscle', 'goals']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing one or more required fields.'}), 400

    try:
        top_experiences = recommender.recommend_coaches(data, all_coaches)
        return jsonify({'recommended_experiences': top_experiences})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import joblib
import xgboost as xgb

class CoachRecommender:
    def __init__(self, model_path='recommender_model_2.pkl'):
        self.model = None
        self.feature_columns = []
        self.skill_list = []
        self.load_model(model_path)

    def load_model(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.skill_list = data['skill_list']

    def prepare_features(self, df):
        data = df.copy()
        features = pd.DataFrame()

        features['trainee_age'] = data['age']
        features['trainee_height'] = data['height']
        features['trainee_weight'] = data['weight']
        features['trainee_body_fat'] = data['body_fat']
        features['trainee_body_muscle'] = data['body_muscle']
        features['trainee_bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
        features['fitness_level'] = data['body_muscle'] - data['body_fat']

        for skill in self.skill_list:
            skill_clean = skill.replace(' ', '_')
            features[f'trainee_wants_{skill_clean}'] = data['trainee_goals'].str.contains(skill, na=False).astype(int)
            features[f'coach_has_{skill_clean}'] = data['coach_experiences'].str.contains(skill, na=False).astype(int)
            features[f'skill_match_{skill_clean}'] = features[f'trainee_wants_{skill_clean}'] * features[f'coach_has_{skill_clean}']

        features['trainee_goal_count'] = data['trainee_goals'].str.count(r'\|') + 1
        features['coach_skill_count'] = data['coach_experiences'].str.count(r'\|') + 1
        features['matching_skills_count'] = data.apply(
            lambda row: len(set(row['trainee_goals'].split('|')) &
                            set(row['coach_experiences'].split('|'))), axis=1
        )
        features['match_percentage'] = features['matching_skills_count'] / features['trainee_goal_count']

        features['age_group_match'] = 0
        features.loc[(data['age'] < 30) & data['coach_experiences'].str.contains('hiit|crossfit', na=False), 'age_group_match'] = 1
        features.loc[(data['age'] >= 45) & data['coach_experiences'].str.contains('flexibility|injury prevention', na=False), 'age_group_match'] = 1

        features['weight_loss_specialist'] = (data['coach_experiences'].str.contains('weight loss|fat loss|cardio', na=False) & (data['body_fat'] > 25)).astype(int)
        features['muscle_gain_specialist'] = (data['coach_experiences'].str.contains('muscle gain|bodybuilding|strength', na=False) & (data['body_muscle'] < 35)).astype(int)

        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[self.feature_columns]

        return features

    def recommend_coaches(self, trainee_data, coaches_df, top_n=1):
        num_coaches = len(coaches_df)
        df = pd.DataFrame({
            'age': [trainee_data['age']] * num_coaches,
            'height': [trainee_data['height']] * num_coaches,
            'weight': [trainee_data['weight']] * num_coaches,
            'body_fat': [trainee_data['body_fat']] * num_coaches,
            'body_muscle': [trainee_data['body_muscle']] * num_coaches,
            'trainee_goals': [trainee_data['goals']] * num_coaches,
            'coach_id': coaches_df['coach_id'],
            'coach_name': coaches_df['coach_name'],
            'coach_rating': coaches_df['coach_rating'],
            'coach_experiences': coaches_df['coach_experiences']
        })

        features = self.prepare_features(df)
        scores = self.model.predict_proba(features)[:, 1]
        df['predicted_score'] = scores

        top_row = df.sort_values(by='predicted_score', ascending=False).head(top_n).iloc[0]
        return top_row['coach_experiences'].split('|')

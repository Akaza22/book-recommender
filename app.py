from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the similarity model
model = joblib.load('book_recommender_model.pkl')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    favorite_books = data.get('favorite_books')

    if not favorite_books:
        return jsonify({'error': 'No favorite books provided'}), 400

    recommended_books = set()
    for book in favorite_books:
        if book in model.index:
            recommended_books.update(model[book].sort_values(ascending=False).head(10).index)

    # Exclude the favorite books from the recommendations
    recommended_books = list(recommended_books - set(favorite_books))
    
    return jsonify({'recommended_books': recommended_books})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

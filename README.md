# 🎬 Hybrid & Explainable Movie Recommendation System

A complete end-to-end Movie Recommendation System built using multiple recommendation paradigms including Collaborative Filtering, Content-Based Filtering, Matrix Factorization (SVD), Neural Collaborative Filtering (Deep Learning), and an Explainability Layer.

This project demonstrates both traditional and modern approaches to recommendation systems using MovieLens-style data.

---

# 📖 Project Overview

The goal of this project is to build a robust and explainable movie recommendation system that:

- Predicts user ratings
- Recommends similar movies
- Combines collaborative and content-based approaches
- Uses deep learning for improved performance
- Provides human-readable explanations for recommendations

This is not a single-model project — it is a full exploration of multiple recommendation strategies.

---

# 📂 Dataset

The system uses MovieLens-style structured data:

## 1️⃣ ratings.csv
- userId
- movieId
- rating
- timestamp

## 2️⃣ movies.csv
- movieId
- title
- genres

---

# ⚙️ Implemented Recommendation Techniques

## 1️⃣ User-Based Collaborative Filtering

- Built using cosine similarity on the user-item sparse matrix
- Recommends movies liked by similar users
- Memory-based collaborative approach

Uses:
- userId
- movieId
- rating

---

## 2️⃣ Item-Based Collaborative Filtering

- Computes similarity between movies based on rating patterns
- Recommends movies similar to a given movie

Uses:
- userId
- movieId
- rating

---

## 3️⃣ Content-Based Filtering (TF-IDF)

- Uses TF-IDF vectorization on movie genres
- Computes cosine similarity between genre vectors
- Recommends movies with similar genre profiles

Uses:
- genres
- title

---

## 4️⃣ Hybrid Recommendation System

Combines:
- Collaborative similarity (ratings-based)
- Content similarity (genre-based)

Final score:
Hybrid Score = α(Content Score) + (1 − α)(Collaborative Score)

This approach improves:
- Cold-start handling
- Robustness
- Recommendation quality

---

## 5️⃣ Matrix Factorization (Truncated SVD)

- Applies dimensionality reduction on the user-item matrix
- Learns latent user and movie embeddings
- Captures hidden preference patterns

This is a model-based collaborative filtering technique.

---

## 6️⃣ Neural Collaborative Filtering (Deep Learning)

Built using TensorFlow / Keras.

Architecture:
- User Embedding Layer
- Movie Embedding Layer
- Fully Connected Dense Layers
- Regression Output (rating prediction)

Loss Function:
- Mean Squared Error (MSE)

Evaluation Metrics:
- RMSE
- MAE
- Rating Accuracy

The neural model achieved strong predictive performance on test data.

---

# 📊 Model Evaluation

Example Test Results:

- Test MSE: 0.8812
- Test RMSE: 0.9270
- Test MAE: 0.7135
- Test Accuracy: 31.37%

RMSE below 1.0 indicates strong rating prediction performance.

---

# 🔍 Explainability Layer

An explainability module was added to generate human-readable explanations for recommendations.

The system explains recommendations by:

- Identifying movies the user rated highly
- Comparing genre overlap
- Selecting the most similar previously liked movie
- Generating explanation text

Example:

"This movie is recommended because you liked 'Toy Story', and both share genres: Animation, Comedy."

This improves:
- Transparency
- Trust
- Interpretability

---

# 🧠 Key Concepts Demonstrated

- Memory-Based Collaborative Filtering
- Model-Based Collaborative Filtering
- Content-Based Recommendation
- Hybrid Systems
- Matrix Factorization
- Embedding Learning
- Deep Learning for Recommendations
- Post-hoc Explainability

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- TensorFlow / Keras

---

# 🚀 How to Run

1. Clone the repository
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the notebook or Python scripts.

---

# 🎯 Why This Project Matters

This project demonstrates practical understanding of:

- Recommendation system architectures
- Similarity metrics
- Latent factor modeling
- Deep learning in recommender systems
- Explainable AI (XAI)

It shows the ability to implement, compare, and evaluate multiple recommendation strategies within one unified system.

---

# 📌 Future Improvements

- Add SHAP-based neural explainability
- Incorporate temporal modeling
- Deploy as a web application
- Use larger dataset for scalability testing

---

# 👤 Author

Muhammad Abbas  
BS Data Science  
Machine Learning & AI Enthusiast

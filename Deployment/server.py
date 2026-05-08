import os
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline, AutoModelForSequenceClassification, RobertaTokenizer

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "sentiment_model")
tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# --- LOAD DATA ---
df_clusters = pd.read_csv(os.path.join(BASE_DIR, "clustered_products.csv"))
df_reviews = pd.read_csv(os.path.join(BASE_DIR, "product_reviews.csv"))
df = pd.merge(df_reviews, df_clusters, on="name", how="inner")

SENT_COL = 'pred_label_name' if 'pred_label_name' in df.columns else 'sentiment'
RATING_COL = 'reviews.rating_x' if 'reviews.rating_x' in df.columns else 'reviews.rating'

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/api/dashboard")
def dashboard():
    # Calculation logic for metrics
    avg_rat = round(float(df[RATING_COL].mean()), 2)
    return jsonify({
        "total": len(df),
        "avg_rating": avg_rat,
        "pos_pct": 70, # Simplified for example
        "neg_pct": 15,
        "neu_pct": 15
    })

@app.route("/api/submit_review", methods=["POST"])
def submit_review():
    global df
    data = request.get_json()
    
    # AI Classification
    text = data.get("text", "")
    prediction = sentiment_pipe(text[:512])[0]
    sentiment_label = prediction['label'].lower()

    # Create new row entry
    new_entry = {
        "name": data.get("product"),
        "cluster_name": data.get("cat"),
        RATING_COL: int(data.get("rating")),
        "reviews.text": text,
        "reviews.username": data.get("name"),
        SENT_COL: sentiment_label
    }
    
    # Add to DataFrame
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    
    return jsonify({
        "status": "success", 
        "sentiment": sentiment_label
    })

if __name__ == "__main__":
    app.run(debug=True, port=5050)
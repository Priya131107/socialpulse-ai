#!/bin/bash
# Streamlit Cloud startup script
# Runs once to set up data and models if not already present

echo "=== SocialPulse AI - Cloud Setup ==="

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# Generate synthetic data if not present
if [ ! -f "data/raw/posts_dataset.csv" ]; then
    echo "Generating synthetic dataset..."
    python -m src.data.synthetic_generator
fi

# Preprocess if splits not present
if [ ! -f "data/processed/train.csv" ]; then
    echo "Preprocessing data..."
    python -m src.data.preprocess
fi

# Train viral model if not present
if [ ! -f "models_saved/best_viral_model.joblib" ]; then
    echo "Training viral prediction model (this may take 2-3 minutes)..."
    python -m src.models.train_viral
fi

# Train crisis detector if not present
if [ ! -f "models_saved/crisis_detector.joblib" ]; then
    echo "Training crisis detector..."
    python -m src.models.train_crisis
fi

echo "=== Setup complete! ==="

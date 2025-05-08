from transformers import pipeline
import pandas as pd

df = pd.read_csv("stream_qos_dataset.csv")
comments = df['viewer_comment'].tolist()

# Use pretrained sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

results = sentiment_pipeline(comments[:20])  # Demo on 20 comments
for comment, result in zip(comments[:20], results):
    print(f"{comment} => {result['label']} ({result['score']:.2f})")

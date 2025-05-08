import pandas as pd
import numpy as np

# Simulate QoS metrics and viewer satisfaction
np.random.seed(42)
data_size = 1000

df = pd.DataFrame({
    'bitrate': np.random.normal(3500, 500, data_size),
    'latency': np.random.normal(200, 50, data_size),
    'frame_drops': np.random.randint(0, 100, data_size),
    'buffering_events': np.random.randint(0, 5, data_size),
    'viewer_comment': np.random.choice([
        "Great stream!", "Too much lag", "Buffering again", "Love this!", "Not smooth", "Excellent quality"
    ], data_size),
})

# Heuristic satisfaction label
df['satisfaction'] = np.where(
    (df['latency'] < 250) & (df['bitrate'] > 3000) & (df['buffering_events'] < 2),
    1,  # Satisfied
    0   # Not satisfied
)

df.to_csv("stream_qos_dataset.csv", index=False)
print("Dataset created.")


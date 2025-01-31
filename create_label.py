import pandas as pd
import os

# Define your audio directory
AUDIO_DIR = 'audio'

# Create a list to store filenames and labels
data = []

# Map of labels based on filename patterns (adjust these based on your actual filenames)
label_map = {
    'heartbeat': 'Normal',
    'sound-effects': 'Artifact',
    'loud': 'Murmur',
    'unlabelled': 'Unlabelled'
}

# Walk through the audio directory
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(('.wav', '.mp3')):  # Add other audio extensions if needed
        # Determine label from filename
        label = 'Unlabelled'  # default label
        for key in label_map:
            if key in filename.lower():
                label = label_map[key]
                break
                
        data.append({
            'filename': filename,
            'label': label
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('labels.csv', index=False)
print("Labels file created successfully!")
print("\nFirst few entries:")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts()) 
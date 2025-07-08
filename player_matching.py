import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# paths
FEATURES_BROADCAST = 'features_broadcast.npy'
FEATURES_TACTICAM = 'features_tacticam.npy'

data_broadcast = np.load(FEATURES_BROADCAST, allow_pickle=True).item()
data_tacticam = np.load(FEATURES_TACTICAM, allow_pickle=True).item()

features_broadcast = data_broadcast['features']
filenames_broadcast = data_broadcast['filenames']

features_tacticam = data_tacticam['features']
filenames_tacticam = data_tacticam['filenames']

similarity_matrix = cosine_similarity(features_broadcast, features_tacticam)

best_matches = similarity_matrix.argmax(axis=1)  # For each broadcast crop, find the best tacticam crop
match_scores = similarity_matrix.max(axis=1)

print("\nPlayer Matching Results:")
for i, (broadcast_file, match_idx, score) in enumerate(zip(filenames_broadcast, best_matches, match_scores)):
    tacticam_file = filenames_tacticam[match_idx]
    print(f"{broadcast_file}  -->  {tacticam_file}  (Score: {score:.4f})")

# Save as mapping 
player_mapping = []
for broadcast_file, match_idx in zip(filenames_broadcast, best_matches):
    tacticam_file = filenames_tacticam[match_idx]
    player_mapping.append((broadcast_file, tacticam_file))

np.save('player_mapping.npy', player_mapping)
print("\nPlayer mapping saved to player_mapping.npy")

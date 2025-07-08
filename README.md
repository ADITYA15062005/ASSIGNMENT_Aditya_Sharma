# ASSIGNMENT_Aditya_Sharma

MODEL DOWNLOAD LINK:-https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
MATERIAL DOWNLOAD LINK:-https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP

➡️OBJECTIVE

Given two videos (broadcast.mp4 and tacticam.mp4) showing the same gameplay from different camera angles, this project identifies and tracks players, extracts visual features, and matches players across both videos using consistent player IDs.

Setup Instructions

1. Clone the repository / Download project folder

git clone <your-repo-link>
cd your-repo-folder

2. Create a Python virtual environment

python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Install model and material videos
   
from the link provided above

➡️Running the Pipeline

1. Player Detection + Tracking

Run this script on both videos:

python save_player_crops.py

Change VIDEO_PATH and OUTPUT_FOLDER in the script for broadcast.mp4 and tacticam.mp4

2. Extract Player Features

Run this for both crops/broadcast and crops/tacticam:

python extract_features.py

Change CROP_FOLDER and FEATURES_SAVE_PATH in the script.

3. Match Players Across Videos

python match_players.py

This will output player matches between the two camera feeds.

➡️Dependencies

Python 3.10+

ultralytics

opencv-python

deep_sort_realtime

torchvision

torch

numpy

tqdm

scikit-learn

Install all dependencies via:

pip install -r requirements.txt


from flask import Flask, request, jsonify, render_template
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from model import Encoder  # Assuming this is the structure of your model.py
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from PIL import Image
import requests
import sqlite3
from urllib.parse import urlparse, parse_qs
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
SPOTIFY_ID = os.environ.get("SPOTIFY_ID")
SPOTIFY_SECRET = os.environ.get("SPOTIFY_SECRET")

# Directory settings
input_folder = 'spotify_downloads'
output_folder = 'outputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Image transform and cropping utility
transform = transforms.Compose([transforms.ToTensor()])
def custom_crop(img):
    return img.crop((0, 16, img.width - 8, img.height))

# Pretrained model
model_path = './encoder.pth'
model = Encoder()  # Assuming Encoder is correctly imported from model.py
model.load_state_dict(torch.load(model_path))
model.eval()

# Spotify API Token
spotify_access_token = None

def authenticate_spotify():
    global spotify_access_token
    url = "https://accounts.spotify.com/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": SPOTIFY_ID,
        "client_secret": SPOTIFY_SECRET
    }
    response = requests.post(url, headers=headers, data=data)
    response_data = response.json()
    spotify_access_token = response_data['access_token']

def get_spotify_song_id(song_link):
    parsed_url = urlparse(song_link)
    if parsed_url.hostname != 'open.spotify.com' or 'track' not in parsed_url.path:
        raise ValueError("Invalid Spotify track URL")
    return parsed_url.path.split('/')[-1]

def fetch_spotify_preview(song_link):
    global spotify_access_token

    # Authenticate if necessary
    if spotify_access_token is None:
        authenticate_spotify()

    # Extract song ID from the link
    song_id = get_spotify_song_id(song_link)
    if not song_id:
        raise ValueError("Could not extract song ID from the link")

    # Fetch song details
    url = f"https://api.spotify.com/v1/tracks/{song_id}?market=US"
    headers = {"Authorization": f"Bearer {spotify_access_token}"}
    response = requests.get(url, headers=headers)
    track_data = response.json()

    # Download the preview
    preview_url = track_data.get('preview_url')
    if not preview_url:
        raise ValueError("No preview URL found for this track")

    preview_response = requests.get(preview_url)
    preview_path = os.path.join(input_folder, f"{song_id}.mp3")
    with open(preview_path, 'wb') as file:
        file.write(preview_response.content)

    return preview_path, preview_url

def process_song(song_link):
    # Fetch and save preview from Spotify
    preview_path, preview_url = None, None

        # Fetch and save preview from Spotify
    preview_path, preview_url = fetch_spotify_preview(song_link)

    # Process the audio file
    y, sr = librosa.load(preview_path, sr=None, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Temporary file path for spectrogram image
    temp_image_path = os.path.join(output_folder, 'temp_mel.png')
    plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()

    # Load and crop image
    img = Image.open(temp_image_path)
    cropped_img = custom_crop(img)

    # Convert image to RGB
    rgb_img = cropped_img.convert('RGB')
    rgb_img.save(temp_image_path)

    # Load image as tensor
    img_as_tensor = transform(rgb_img)

    # Pass through the model
    model_output = model(img_as_tensor.unsqueeze(0))

    model_output_list = model_output.cpu().detach().numpy().flatten().tolist()

    # Make Qdrant API call
    client = QdrantClient(url="https://db.ncdedinsky.com", port=443, api_key=QDRANT_API_KEY)
    index_name = 'song-embeddings-index'

    response = client.search(
        collection_name=index_name,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        query_vector=model_output_list,
        limit=100,
    )

    ids = [point.payload['spotify_id'] for point in response]

    if ids[0] == get_spotify_song_id(song_link):
        ids = ids[1:]

    # this should be changed later
    ids = list(filter(lambda id: len(id) == 22, ids))

    ids = ids[:2]

    # Initialize list to hold song data
    songs_data = []
            #{"id": "given_song", "url_spotify_preview": preview_url, "artist" : "you:", "name" : "Provided"}]

    # Connect to SQLite database
    conn = sqlite3.connect('metadata.db')
    cursor = conn.cursor()

    # Fetch data for each ID
    for id in ids:
        cursor.execute("SELECT preview_url, artist_name, song_name FROM songs WHERE spotify_id = ?", (id,))
        data = cursor.fetchone()
        if data:
            songs_data.append({
                'id': id,
                'url_spotify_preview': data[0],
                'artist': data[1],
                'name': data[2],
                'condition': 'treatment'
            })

    # Close the database connection
    conn.close()

    # Clean up downloaded files
    os.remove(temp_image_path)

    # Return the songs data
    return songs_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_song', methods=['POST'])
def process_song_endpoint():
    song_link = request.json.get('song_link')
    if not song_link:
        return jsonify({'error': 'No song link provided'}), 400

    return jsonify(process_song(song_link))

@app.route('/recommend_song', methods=['POST'])
def recommend_song():
    # Your existing POST handling code
    song_link = request.json.get('song_link')
    if not song_link:
        return jsonify({'error': 'No song link provided'}), 400

    try:
        # Fetch and process the provided song
        process_result = process_song(song_link)
        recommended_songs = process_result

        # Fetch two random songs from the SQLite database
        conn = sqlite3.connect('metadata.db')
        cursor = conn.cursor()
        cursor.execute("SELECT spotify_id, preview_url, artist_name, song_name FROM songs ORDER BY RANDOM() LIMIT 2")
        random_songs_data = cursor.fetchall()
        conn.close()

        random_songs = [
            {
                'id': song[0],
                'url_spotify_preview': song[1],
                'artist': song[2],
                'name': song[3],
                'condition': 'control'
            } for song in random_songs_data
        ]

        # Combine recommended and random songs
        result_songs = recommended_songs + random_songs

        #print(result_songs)
        return jsonify(result_songs)

    except Exception as e:
        #print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)

# How to use this project

### Relevant folders
* `downloading_data/`
* `project/`
* `database/`
* `webapp/`

### `downloading_data`
* Holds the scripts and `venv` to download and process `mp3` files into mel spectrogram images. 
    * `converter.py`
        * This converts `mp3`s to mel spectrogram images
    * `downloader.py`
        * This is pretty obvious, it downloads `mp3` files based on the files in `data.db` that are set to `downloaded = 0` and then their values are updated

### `project`
* Holds the source code for training the model, converting images to embeddings, and uploading embeddings to the database
    * `encoder.py`
        * This encodes images into embeddings that can be compared
    * `main.py`
        * Main training runner. This runs the model training based on the imported model
    * `models/`
        * Holds the source code for the different sized models with differing outputs

### `database`
* Holds the `Qdrant` database file directory, start and stop scripts for the database and the database config file which contains the Qdrant API key

### `webapp`
* Holds the code for the simple FE/BE that serves audio to the user. This relies on the metadata.db which is sym-linked to the `data.db` in `downloading_data`

### Necessary components to run the entire thing:
1. Database needs to be running
    * `database/start.sh`
2. Webapp needs to be running
    * `webapp/start_server.sh`

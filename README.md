# How to use this project
> For vector database and configuration, email [me](mailto:dedin005@umn.com), I didn't want to upload all the embeddings to github

### Relevant folders
* `project/`
* `database/`
* `webapp/`

### `project/src/recommender`
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
* Holds the code for the simple FE/BE that serves audio to the user. This relies on the metadata.db which is symlinked to the `data.db` in `downloading_data` (not symlinked on git)

### Necessary components to run the entire thing:
1. Database needs to be running
    * `database/start.sh`
2. Webapp needs to be running
    * `webapp/start_server.sh`

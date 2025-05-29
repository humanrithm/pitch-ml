import os
import dropbox
from dotenv import load_dotenv

def connect_to_dropbox():
    secrets_path = os.path.join(os.path.dirname(__file__), 'secrets_box.env')
    load_dotenv(secrets_path)                                                   # load dropbox credentials into environment
    dbx = dropbox.Dropbox(os.getenv('ACCESS_TOKEN'))                            # connect to dropbox (NOTE: have to refresh token frequently)

    return dbx
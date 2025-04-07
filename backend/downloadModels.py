'''Before running the below code make a sure the token is correct, you can create a new token if u ont have one from --> https://huggingface.co/settings/tokens'''
#! make a .env file and add a variable named "HUGGINGFACE_TOKEN" before running
import dotenv
from huggingface_hub import snapshot_download
import os

dotenv.load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
models = [ 
    "stabilityai/stable-diffusion-xl-base-1.0",
    "mistralai/Mistral-7B-Instruct-v0.3"
]
#! if the download_path is set to None then it will directly go to --> C:\Users\<YOUR_NAME>\.cache\huggingface\hub 
download_path = r"E:\models"

for model in models:
    print(f"downloading {model}...")
    snapshot_download(repo_id=model,
                      cache_dir=download_path,
                      token= huggingface_token)
    print(f"Downloaded the model --> {model}")
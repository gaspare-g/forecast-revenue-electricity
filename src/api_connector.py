from dotenv import load_dotenv
import kagglehub

class APIConnector:

    def __init__(self):
        load_dotenv()

    def connect_to_kaggle(self):
        path = kagglehub.dataset_download("atharvasoundankar/global-energy-consumption-2000-2024")
        print("Path to dataset files:", path)
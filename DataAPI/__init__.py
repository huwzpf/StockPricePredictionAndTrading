import nasdaqdatalink
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
nasdaqdatalink.ApiConfig.api_key = api_key

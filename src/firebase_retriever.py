import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from yaml import safe_load
import json
import pandas as pd

# Load the model_link from the config.yaml file
with open('src/config.yaml', 'r') as yaml_file:
    config = safe_load(yaml_file)
model_link_to_filter = config.get('model_use', None)
firebase_database_url = config.get('firebase_database_url', None)

if model_link_to_filter is None:
    raise ValueError("The 'model_link' field is missing in config.yaml.")

cred = credentials.Certificate('secrets/service-account.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': firebase_database_url
})

ref = db.reference('/')
data = ref.get().get("trainingMetrics")

df = pd.DataFrame(data).transpose()
df = df.where(df["modelLink"] == model_link_to_filter).dropna()
df = df.where(df["sampleEnergy"] > 0 ).dropna()

df.to_json("deviceConfigurations.json", orient="records", indent=4)
print("Filtered data exported successfully!")

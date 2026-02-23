import os
import sys
from pathlib import Path

# Add project root to python path to allow src imports
sys.path.append(str(Path(__file__).parent.parent))

print("Attempting to import config...")
try:
    from src import config
    print("Config imported.")
except Exception as e:
    print(f"Config import failed: {e}")
    sys.exit(1)

from huggingface_hub import whoami

print("Checking Hugging Face identity...")
try:
    info = whoami()
    print(f"Logged in as: {info.get('name', 'Unknown')}")
    print(f"Orgs: {[org['name'] for org in info.get('orgs', [])]}")
    print(f"Auth Method: {info.get('auth_type')}")
except Exception as e:
    print(f"Not logged in or error: {e}")

from huggingface_hub import model_info

models_to_check = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct"
]

print("\nChecking model access:")
for model_id in models_to_check:
    print(f"Checking {model_id}...", end=" ")
    try:
        model_info(model_id)
        print("ACCESS GRANTED ✅")
    except Exception as e:
        print(f"ACCESS DENIED ❌ ({e})")

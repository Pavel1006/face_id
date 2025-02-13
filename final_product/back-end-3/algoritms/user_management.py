import json

USERS_FILE = "users_data.json"

def load_users():
    """Load users from the JSON file."""
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_users(users_data):
    """Save users data to the JSON file."""
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users_data, f, indent=4)
    except Exception as e:
        print(f"Error saving users data: {str(e)}")

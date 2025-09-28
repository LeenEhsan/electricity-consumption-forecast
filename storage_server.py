from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# Where uploaded files are stored
UPLOAD_FOLDER = "uploads"
BASE_URL = "https://your-storage-server.com/uploads"

@app.route("/api/files", methods=["GET"])
def list_user_files():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "Missing user parameter"}), 400

    user_dir = os.path.join(UPLOAD_FOLDER, user)
    if not os.path.isdir(user_dir):
        return jsonify({"files": []})  # No files yet

    file_entries = []
    for filename in os.listdir(user_dir):
        filepath = os.path.join(user_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = filename.split(".")[-1].lower()
        file_stat = os.stat(filepath)
        created = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d")

        file_entries.append({
            "name": filename,
            "type": ext,
            "date": created,
            "url": f"{BASE_URL}/{user}/{filename}"
        })

    return jsonify({"files": file_entries})

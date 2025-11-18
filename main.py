import os
import json
import io
from flask import Flask, request, jsonify
from google.cloud import storage
import torch
from safetensors import safe_open

app = Flask(__name__)

# -----------------------
# Configuration
# -----------------------
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "ain-model")  # GCS bucket for shards
INDEX_FILE = os.getenv("INDEX_FILE", "model.safetensors.index.json")  # Local index file

# -----------------------
# Load index JSON
# -----------------------
with open(INDEX_FILE, "r") as f:
    weight_map = json.load(f)

# Replace local shard paths with GCS paths if needed
for key, path in weight_map["weight_map"].items():
    if not path.startswith("gs://"):
        weight_map["weight_map"][key] = f"gs://{MODEL_BUCKET}/{path}"

# -----------------------
# Helper: Load tensor on-the-fly
# -----------------------
def load_tensor_from_gcs(param_path, param_name):
    """
    Load a single tensor from a GCS shard.
    Supports safetensors with remote streaming if available.
    """
    if param_path.startswith("gs://"):
        try:
            # Try using safetensors remote streaming first
            with safe_open(param_path, framework="pt", remote=True) as f:
                return f.get_tensor(param_name)
        except Exception:
            # Fallback to GCS client + in-memory
            bucket_name, blob_name = param_path[5:].split("/", 1)
            client = storage.Client()
            blob = client.bucket(bucket_name).blob(blob_name)
            data = blob.download_as_bytes()
            buffer = io.BytesIO(data)
            # torch.load can be used if tensors are saved as PyTorch tensors
            return torch.load(buffer)
    else:
        # Local path fallback
        return torch.load(param_path)

# -----------------------
# Example endpoint
# -----------------------
@app.route("/get_tensor", methods=["GET"])
def get_tensor():
    """
    Example endpoint:
    GET /get_tensor?param_name=model.embed_tokens.weight
    """
    param_name = request.args.get("param_name")
    if param_name not in weight_map["weight_map"]:
        return jsonify({"error": f"Parameter {param_name} not found"}), 404

    param_path = weight_map["weight_map"][param_name]
    tensor = load_tensor_from_gcs(param_path, param_name)
    
    # Return basic info about tensor
    return jsonify({
        "param_name": param_name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype)
    })


# -----------------------
# Run Flask app (Cloud Run)
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)

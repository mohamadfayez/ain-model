import requests
import json

URL = "http://localhost:8080/infer"

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/MBZUAI/AIN/resolve/main/assets_hf/demo_image.jpeg"
                },
                {
                    "type": "text",
                    "text": "يرجى وصف هذه الصورة."
                }
            ]
        }
    ]
}

response = requests.post(URL, json=payload)

print("\n=== Server Response ===")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

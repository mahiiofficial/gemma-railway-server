from flask import Flask, request, jsonify
import os
import torch
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

tts = Text2Speech.from_pretrained(
    model_tag="kan-bayashi/tts_urtts_train_raw_phn_tacotron2",
    device="cpu"
)

@app.route("/")
def home():
    return "✅ Urdu TTS Server Running"

@app.route("/speak", methods=["POST"])
def speak():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    with torch.no_grad():
        wav, *_ = tts(text)
        out_path = "static/output.wav"
        sf.write(out_path, wav.view(-1).cpu().numpy(), tts.fs)

    return jsonify({"audio_url": request.url_root + "static/output.wav"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

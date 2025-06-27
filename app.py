from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model once
print("Loading Gemma model... please wait")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it")
print("Gemma loaded ✅")

@app.route("/gemma", methods=["POST"])
def gemma_chat():
    data = request.get_json()
    prompt = data.get("text", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"reply": reply})

@app.route("/", methods=["GET"])
def health():
    return "Gemma Server Running ✅"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

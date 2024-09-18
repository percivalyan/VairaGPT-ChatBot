from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the pre-trained GPT-2 model and tokenizer
MODEL_NAME = "gpt2"
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Configuration parameters
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2
INAPPROPRIATE_WORDS = ["shit", "crap", "damn"]
STOP_TOKENS = ['.', '!', '?']  # Define possible stop tokens
DEFAULT_MAX_LENGTH = 400

def adjust_max_length(input_text):
    word_count = len(input_text.split())
    
    if word_count <= 2:
        return 50  # Short response for 1-2 words
    elif word_count <= 5:
        return 100  # Medium response for 3-5 words
    else:
        return DEFAULT_MAX_LENGTH  # Longer response for more than 5 words

def generate_response(input_text):
    try:
        max_length = adjust_max_length(input_text)
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Ensure the response ends with a stop token
        if not any(generated_text.endswith(stop_token) for stop_token in STOP_TOKENS):
            output = model.generate(
                input_ids,
                max_length=max_length + 100,  # Increase length if needed
                num_return_sequences=1,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Check for inappropriate content
        if any(word in generated_text.lower() for word in INAPPROPRIATE_WORDS):
            return "Sorry, the response contains inappropriate content. Please try another input."

        return generated_text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    print(f"User input: {user_input}")  # Debug print
    if user_input:
        bot_response = generate_response(user_input)
        print(f"Bot response: {bot_response}")  # Debug print
        return jsonify({"response": bot_response})
    return jsonify({"response": "Please enter a valid message."})

if __name__ == "__main__":
    app.run(debug=True)

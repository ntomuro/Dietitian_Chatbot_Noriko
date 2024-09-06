from flask import Flask, request, jsonify
from ragm import ask  # Import the RAG chatbot

app = Flask(__name__)

@app.route('/get_chat_response', methods=['POST'])
def get_chat_response():
    msg = request.form.get('msg')
    if msg:
        response = ask(msg)
        return jsonify({'response': response})
    return jsonify({'error': 'No message received'})

if __name__ == "__main__":
    app.run(debug=True)

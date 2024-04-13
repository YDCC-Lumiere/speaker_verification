from flask import Flask, request, jsonify
import os

app = Flask(name)

UPLOADFOLDER = 'temp'
if not os.path.exists(UPLOADFOLDER):
    os.makedirs(UPLOADFOLDER)

@app.route('/upload', methods=['POST'])
def get_embed():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    data = ...

    return jsonify({'embed': data}), 200

if __name == '__main':
    app.run(debug=True)
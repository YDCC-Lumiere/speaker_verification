import os

from flask import Flask, request, jsonify

from model.titanet import TitaNet

app = Flask(__name__)

UPLOADFOLDER = 'verify'
if not os.path.exists(UPLOADFOLDER):
    os.makedirs(UPLOADFOLDER)

model = TitaNet

@app.route('/verify', methods=['POST'])
def get_embed():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = os.path.join(UPLOADFOLDER, file.filename)
    file.save(filename)

    data = model.verify(filename)

    return jsonify({'verify': data}), 200


if __name__ == '__main__':
    app.run(debug=True)

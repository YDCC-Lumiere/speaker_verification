import time
import os

from flask import Flask, request, jsonify

from model.titanet import TitaNet, verify_embs

app = Flask(__name__)

UPLOADFOLDER = 'temp'
if not os.path.exists(UPLOADFOLDER):
    os.makedirs(UPLOADFOLDER)

model = TitaNet.load_from_checkpoint('../ck_1.ckpt')

@app.route('/upload', methods=['POST'])
def get_embed():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = os.path.join(UPLOADFOLDER, file.filename)
    file.save(filename)
    print(filename)

    norm_filename = 'temp/norm_audio.wav'
    os.system(f'ffmpeg -loglevel panic -y -i {filename} -ar 16000 -ab 256000 -ac 1 {norm_filename}')

    data = model.get_avg_embedding(audio=norm_filename)

    return jsonify({'embed': data}), 200


@app.route('/verify', methods=['POST'])
def verify():
    data = request.json

    if 'original' not in data or 'verify' not in data:
        return jsonify({'error': 'Missing data in request'}), 400

    original_emb = data['original']
    verify_emb = data['verify']

    data = verify_embs(original_emb, verify_emb)

    return jsonify({'verify': data}), 200


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------
# Load models & tokenizer
# --------------------------
feature_extractor = tf.keras.models.load_model('feature_extractor.keras')
decoder_model = tf.keras.models.load_model('model.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 34  # as trained

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def generate_caption(image_features):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = decoder_model.predict([np.expand_dims(image_features, axis=0), sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').replace('endseq', '').strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            img_features = feature_extractor.predict(img, verbose=0)
            if img_features.ndim > 1:
                img_features = img_features[0]

            caption = generate_caption(img_features)
            filename = file.filename

    return render_template('index.html', caption=caption, filename=filename)

# Endpoint for demo images
@app.route('/use-demo/<filename>', methods=['GET'])
def use_demo(filename):
    src = os.path.join('static/demo', filename)
    dest = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
        fdst.write(fsrc.read())

    img = preprocess_image(dest)
    img_features = feature_extractor.predict(img, verbose=0)
    if img_features.ndim > 1:
        img_features = img_features[0]

    caption = generate_caption(img_features)
    return render_template('index.html', caption=caption, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

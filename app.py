from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Initialisation de l'application Flask
app = Flask(__name__)

#Chargement du modèle pré-entrainé
model = load_model('model_incendie.h5')

#Les classes à prédire
classes = ['Non enflammé', 'Enflammé']

#Définition du dossier pour le téléchargement des images
UPLOAD_FOLDER = os.path.join(app.root_path, 'jeu_de_données_images')
#UPLOAD_FOLDER = os.path.join('/jeu_de_données_images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#LEs extensions d'images autorisées
allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

#Fonction de verification si l'extension a été respectée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in allowed_extensions

#Route pour la page d'accueille
@app.route('/')
def index():
    return render_template('index.html')

#fonction pour prédire l'image téléchargée
#@app.route('/jeu_de_données_images', methods = ['GET','POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('result.html', prediction=prediction, filename=filename)
    else:
        return redirect(request.url)
    
#Predict the image using model
def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /=255.0
    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]
    return result

if __name__== "__main__":
    app.run(debug=True)




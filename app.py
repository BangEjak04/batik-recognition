from flask import Flask, request, jsonify, render_template, url_for
from datetime import datetime
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

dic = {0: 'Batik Dayak', 1: 'Batik Geblek Renteng', 2: 'Batik Ikat Celup', 3: 'Batik Insang', 4: 'Batik Kawung', 5: 'Batik Lasem', 6: 'Batik Megamendung', 7: 'Batik Pala', 8: 'Batik Parang', 9: 'Batik Sekar Jagad', 10: 'Batik Tambal'}

model = load_model('Cendekia-Batik Exception-77.04.keras')

model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = i.reshape(1, 224,224,3)
    # Get the prediction probabilities
    predictions = model.predict(i)
    # Get the class with the highest probability
    predicted_class = predictions.argmax(axis=-1)[0]  # Take the index of the max probability
    return dic[predicted_class]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if 'my_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    img = request.files['my_image']

    # Simpan gambar di folder static
    # img_path = os.path.join("static", img.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.{img.filename.split('.')[-1]}"
    img_path = os.path.join("static", filename)
    img.save(img_path)

    # Prediksi label berdasarkan gambar
    prediction = predict_label(img_path)

    # Kembalikan hasil dalam bentuk JSON
    response = {
        "prediction": prediction,
        "img_path": url_for('static', filename=img.filename)
    }
    return jsonify(response)

if __name__ =='__main__':
	app.run(host='0.0.0.0', port=5001, debug=True)
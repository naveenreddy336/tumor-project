import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.applications import DenseNet121, ResNet101

app = Flask(__name__)
app.secret_key = "secret123"
app.config['UPLOAD_FOLDER'] = 'static'

# Load Models
densenet = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
resnet = ResNet101(weights='imagenet', include_top=False, pooling='avg')

model = joblib.load("tumor_model.pkl")

# ---------- Prediction Function ----------
def predict_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    f1 = densenet.predict(img,verbose=0)
    f2 = resnet.predict(img,verbose=0)

    fused = np.concatenate((f1[0],f2[0]))

    prediction = model.predict([fused])
    probability = model.predict_proba([fused])

    tumor_percent = round(probability[0][1]*100,2)

    if prediction[0] == 1:
        result = "Tumor Detected"
        status = "⚠ Danger"
    else:
        result = "Normal"
        status = "✔ Safe"

    return result, tumor_percent, status


# ---------- Login Page ----------
@app.route("/login",methods=["GET","POST"])
def login():

    if request.method=="POST":

        username=request.form["username"]
        password=request.form["password"]

        if username=="admin" and password=="1234":

            session["user"]=username
            return redirect(url_for("index"))

        else:
            return render_template("login.html",error="Invalid Login")

    return render_template("login.html")


# ---------- Main Page ----------
@app.route("/",methods=["GET","POST"])
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method=="POST":

        file=request.files["file"]

        filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)

        file.save(filepath)

        result,percent,status=predict_image(filepath)

        return render_template("index.html",
                               result=result,
                               percent=percent,
                               status=status,
                               image=filepath)

    return render_template("index.html",result=None)


# ---------- Logout ----------
@app.route("/logout")
def logout():

    session.pop("user",None)

    return redirect(url_for("login"))


if __name__=="__main__":
    app.run(debug=True)
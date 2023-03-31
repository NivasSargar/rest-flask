from flask import Flask, render_template, request
from transformers import pipeline

# from keras.models import load_model
# from keras.preprocessing import image

app = Flask(__name__)




def predict_label(img_path):
    pipe = pipeline(task="image-classification", 
                model="microsoft/dit-base-finetuned-rvlcdip")

    p=pipe(img_path)
    predict=p[0]
    p=predict
    return p
    
    


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename 
        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction = p["label"], score=round((p["score"]*100),2),img_path = img_path)


if __name__ =='__main__':
    #app.debug = True
    app.run(debug = True)
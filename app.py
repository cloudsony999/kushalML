from flask import Flask, render_template,request
import pandas as pd
app = Flask(__name__)


@app.route('/predict')
def call():
    model = pd.read_pickle('kushal.pickle')
    gre = int(request.args.get("n1"))
    tof = int(request.args.get("n2"))
    cgpa = float(request.args.get("n3"))
    # Predict chances
    result = model.predict([[gre, tof, cgpa]])
    return f"Chances are : {result[0] * 100:.2f}%"


@app.route('/')
def homepage():
    return render_template('home.html')


if __name__ == '__main__':
    app.run()

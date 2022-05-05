from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
from wtforms.validators import NumberRange
import numpy as np
from tensorflow.keras.models import load_model
import joblib

#define function for returning the model's prediction
def return_prediction(model, scaler, sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len, s_wid, p_len, p_wid]]
    flower = scaler.transform(flower)

    classes = np.array(['setosa', 'versicolor', 'virginica'])
    class_ind = model.predict_classes(flower)

    return classes[class_ind][0]

#defining the app routes and completing the app.py file
app = Flask(__name__)
#configure the secret key for the session
app.config['SECRET_KEY'] = 'mysecretkey'

#load the model and scaler
model = load_model('iris_model.h5')
scaler = joblib.load('scaler.pkl')

#define the form
class IrisForm(FlaskForm):
    sepal_length = TextField('Sepal Length', validators=[NumberRange(min=0, max=10)])
    sepal_width = TextField('Sepal Width', validators=[NumberRange(min=0, max=10)])
    petal_length = TextField('Petal Length', validators=[NumberRange(min=0, max=10)])
    petal_width = TextField('Petal Width', validators=[NumberRange(min=0, max=10)])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    #create the form
    form = IrisForm()
    #if the form is submitted
    if form.validate_on_submit():
        #store the values in the session
        session['sepal_length'] = form.sepal_length.data
        session['sepal_width'] = form.sepal_width.data
        session['petal_length'] = form.petal_length.data
        session['petal_width'] = form.petal_width.data
        #redirect to the prediction page
        return redirect(url_for('prediction'))
    #if the form is not submitted
    else:
        #return the form
        return render_template('index.html', form=form)

@app.route('/prediction')
def prediction():
    #if the session is empty
    if not session:
        #redirect to the index page
        return redirect(url_for('index'))
    #if the session is not empty
    else:
        #get the values from the session
        s_len = session['sepal_length']
        s_wid = session['sepal_width']
        p_len = session['petal_length']
        p_wid = session['petal_width']
        #create the sample json
        sample_json = {'sepal_length': s_len, 'sepal_width': s_wid, 'petal_length': p_len, 'petal_width': p_wid}
        #get the prediction
        prediction = return_prediction(model, scaler, sample_json)
        #return the prediction
        return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
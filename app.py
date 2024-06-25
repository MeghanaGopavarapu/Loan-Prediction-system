from flask import Flask,url_for
import numpy as np
from flask import request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        l=[]
        int_features=[] 
        gender=request.form['Gender']
        l.append(gender)
        if gender=='Male':
            gender=1
        else:
            gender=0
        int_features.append(int(gender))
        
        married=request.form['Married']
        l.append(married)
        if married=='Yes':
            married=1
        else:
            married=0
        int_features.append(int(married))
        
        education=request.form['Education']
        l.append(education)
        if education=='Graduate':
            education=1
        else:
            education=0
        int_features.append(int(education))
        
        self_employed=request.form['Self-Employed']
        l.append(self_employed)
        if self_employed=='Yes':
            self_employed=1
        else:
            self_employed=0
        int_features.append(int(self_employed))
        
        coapplicantincome=request.form['CoapplicantIncome']
        l.append(coapplicantincome)
        int_features.append(int(coapplicantincome))
        
        loanamountterm=request.form['Loan_Amount_Term']
        l.append(loanamountterm)
        int_features.append(int(loanamountterm))
        
        credithistory=request.form['Credit_History']
        l.append(credithistory)
        int_features.append(int(credithistory))
        
        print(int_features)
        final_features = [np.array(int_features)]
        print(final_features)
        prediction = model.predict(final_features)
        print(prediction)
        
        output = int(prediction)
        print(output)
        
        if output == 0:
            return render_template('form1.html',prediction_text="Nay!!You are not eligible for Loan")
        else:
            return render_template('form1.html',prediction_text='Yayy!!You are eligible for Loan')

if __name__=="__main__":
    app.run(debug=True,use_reloader=False)

        
        
        
        
        
        
        
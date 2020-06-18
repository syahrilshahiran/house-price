import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('C:\\Users\\C0MM@ND3RR0R\\fyphouse\\test\\xgb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
 
	setlocation = ['Ampang','Bandar Saujana Putra','Bangi','Banting','Batu Caves','Beranang','Bukit Beruntung','Cheras','Cyberjaya','Dengkil','Glenmarie','Gombak','Hulu Langat','Hulu Selangor','Jenjarom','Kajang','Kapar','Kayu Ara','Kerling','Klang','Kota Kemuning','Kuala Kubu Baru','Kuala Langat','Kuala Selangor','Petaling Jaya','Port Klang','Puchong','Rawang','Sabak Bernam','Salak Tinggi','Selayang','Sepang','Serdang','Serendah','Shah Alam','Subang Jaya','Sungai Buloh','Tanjong Karang','Tanjong Sepat','Ulu Klang']

	setcategory=['Apartment','Bungalow','Condo','Double storey','One and a half storey','Semi detached','Single storey','Townhouse','Triple storey','Two and a half storey']
	
	cat = request.form.get('property')
	category = pd.Categorical(cat, categories = setcategory)
	catdummies = pd.get_dummies(category)
	catdummies = catdummies.to_string(header=False,index=False)
	
	catstrip = catdummies.strip()
	catsplit = catstrip.split('  ')
	cat_features = [int(x) for x in catsplit]
	
	loc = request.form.get('location')
	location = pd.Categorical(loc, categories = setlocation)
	locdummies = pd.get_dummies(location)
	locdummies=locdummies.to_string(header=False,index=False)
	locstrip = locdummies.strip()
	locsplit = locstrip.split('  ')
	loc_features = [int(x) for x in locsplit]
	
	size = request.form.get('size')
	bath = request.form.get('bathroom')
	bed = request.form.get('bedroom')
	new =[size,bed,bath]
	#int_features = [int(x) for x in request.form.values()]
	#final_features = [np.array(int_features)]
	final_features = np.asarray(new + loc_features + cat_features)
	final_features= final_features.reshape((1,-1))
	prediction = model.predict(final_features)
	output = '%.1f' %round(prediction[0], 1)
	
	
	
	return render_template('index.html', prediction_text='House Price should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

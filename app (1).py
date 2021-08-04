import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from flask import Flask , request, render_template, flash, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("Brain_Tumor5.h5")

@app.route('/')
def index():
    return render_template('mri_ui.html')

@app.route('/mri_redirect',methods = ['POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['image']
        
        #print("current path")
        basepath = os.path.dirname(__file__)
        #print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (256,256))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict(x)
        
        print("prediction",preds)
       
        if(preds>0.5):
            text=" The MRI scan indicates presence of a brain tumor. "
        else:
            text=" The MRI scan does not indicate presence of a brain tumor. "
        
        return render_template('mri_redirect.html', prediction_text=text)

if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    


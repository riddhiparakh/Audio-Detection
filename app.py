from flask import Flask,render_template,request,redirect
import pickle
import audiosegment
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

pickle_in=open("ann.pkl","rb")
ann=pickle.load(pickle_in)

pickle_in=open("pca.pkl","rb")
pca=pickle.load(pickle_in)

app=Flask(__name__)

person={0:'Anisha',1:'Anushkaa',2:'Ayush',3:'Mumma',4:'Nihar',5:'Papa',6:'Pranya',7:'Samarth',8:'Yash'}

@app.route("/",methods=["GET","POST"])
def main():
  if request.method=="POST":
    print("Form data received ")
    
    if "file" not in request.files:
      return redirect(request.url)
    
    file=request.files["file"]
    if file.filename=="":
      return redirect(request.url)
    
    if file:
      pass
      #  load_audio(file)
      #  x=get_arrays("static/image")
      #  pc=scaling(x)
      #  prediction=np.argmax(ann.predict(pc.reshape(1,-1)))
       
  return render_template("index.html")
  

if __name__=="__main__":
  app.run(debug=True,threaded=True)


def load_audio(file):
    seg = audiosegment.from_wav(file)
    freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=0.5)
    amplitudes = 10 * np.log10(amplitudes + 1e-9)
    plt.pcolormesh(times, freqs, amplitudes)
    plt.savefig('static/image')

def load_img(folder):
  img=[]
  for i in os.listdir(folder):
    img1=cv2.imread(os.path.join(folder,i))
    img1=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    img1=cv2.resize(img1,(256,256))
    img.append(img1)
    
  return img

def get_arrays(folder):
  arrays=np.array(load_img(folder))
  # arrays=np.reshape(arrays,(arrays.shape[0],arrays.shape[1]*arrays.shape[2]))
  arr=[]
  for i in range(len(arrays)):
    arr.append(arrays[i].reshape(-1,1))
  arr=np.squeeze(arr)

  return np.array(arr)


def scaling(x):
  from sklearn.preprocessing import StandardScaler
  se=StandardScaler()
  x=se.fit_transform(x)
  
  pc=pca.transform(x)
  return pc





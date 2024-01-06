import numpy as np
import pickle
import pandas as pd
import streamlit as st 


pickle_in=open("ann.pkl","rb")
ann=pickle.load(pickle_in)

def welcome():
  return "Welcome All"


def main():
  st.title("Person classification using audio")
  html_temp="""
           <div style="background-color:blue;padding:10px">
           <h2 style="color:white">Streamlit person detetction app></h2>
           </div>
            """
  st.markdown(html_temp,unsafe_allow_html=True)
  audio=st.audio_input("")
  # html_temp="""
  #   <form method="POST" enctype="multipart/form-data">
  #   <input type="file" name="file">
  #   <br>
  #   <input type="submit" name="submit">
  #  </form>
  # """
  # if st.button("Predict"):
  #   result=
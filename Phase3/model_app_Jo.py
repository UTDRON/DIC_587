import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
import pandas as pd

#taken from https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524
st.set_option('deprecation.showPyplotGlobalUse', False)
def getSubjectivity(text):

   return TextBlob(str(text)).sentiment.subjectivity
  
 #Create a function to get the polarity
def getPolarity(text):
   return TextBlob(str(text)).sentiment.polarity

# Streamlit app title
st.title("Movie Prediction 🎥")
st.header("Explorartory Data Analysis for Movie Dataset")
st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
dataset = st.file_uploader(label = '')

if dataset:
    df = pd.read_csv(dataset)
    st.subheader('Dataframe:')
    n, m = df.shape
    st.write(f'<p style="font-size:130%">Dataset contains {n} rows and {m} columns.</p>', unsafe_allow_html=True)   
    st.dataframe(df)
    st.subheader("Descriptive analysis")
    st.dataframe(df.describe())
    st.subheader("Distribution of Sentiment Score")
    plt.hist(df["sentiment"], linewidth = 0.5, bins=30)
    plt.title("Distribution of Sentiment Score")
    plt.xlabel("Sentiment score")
    plt.ylabel("Count")
    st.pyplot(plt.show())
    st.subheader("Distribution of Subjective Score")
    plt.hist(df["subjective"], linewidth = 0.5, bins=15, log =True)
    plt.xlabel("Subjectivity score")
    plt.title("Distribution of Subjectivity Score")
    plt.ylabel("Count")
    st.pyplot(plt.show())
    df['release_date'].isna().sum()
    x=df[df['release_date'].isna()]['index']
    df=df.drop(x)
    df['year'] = pd.DatetimeIndex(df['release_date']).year
    df['month'] = pd.DatetimeIndex(df['release_date']).month
    df['year'] = df['year'].astype(int)
    df['month']=df['month'].astype(int)
    y_point= []
    for x in range(1,13):
        y_point.append(df.groupby("month").get_group(x).popularity.mean())

    Label= ["January", "February","March","April","May","June","July", "August","September","October","November","December"]
    x_point = [1,2,3,4,5,6,7,8,9,10,11,12]
    st.subheader("Trend of Popularity of Movies Released in Specific Months")
    plt.plot(x_point, y_point)
    plt.xticks(x_point, Label, rotation='vertical')
    plt.title("Trend of Popularity of Movies Released in Specific Months")
    plt.xlabel("Months")
    plt.ylabel("Popularity")
    st.pyplot(plt.show())
    st.subheader("Scatter Plot Revenue vs Poplarity")
    y = np.log(df.revenue, dtype='float64')
    x = np.log(df.popularity, dtype='float64')
    plt.scatter(x = x, y= y, alpha= 0.9, s= 2)
    plt.title("Scatter Plot Revenue vs Poplarity")
    plt.xlabel("Popularity")
    plt.ylabel("Revenue")
    st.pyplot(plt.show())

    st.subheader("Trend of Average Revenue made by Movies Released in each Month")
    y_point= []
    for x in range(1,13):
        y_point.append(df.groupby("month").get_group(x).revenue.mean())

    Label= ["January", "February","March","April","May","June","July", "August","September","October","November","December"]
    x_point = [1,2,3,4,5,6,7,8,9,10,11,12]
    plt.plot(x_point, y_point)
    plt.xticks(x_point, Label, rotation='vertical')
    plt.title("Trend of Average Revenue made by Movies Released in each Month")
    plt.xlabel("Months")
    plt.ylabel("Revenue")
    st.pyplot(plt.show())

# Visualization pane at the top
st.header("Popularity and Revenue prediction")
loaded_model =  pickle.load(open("trained_model_popularity.sav", "rb"))
loaded_model_revenue =  pickle.load(open("trained_model_revenue.sav", "rb"))
scaler = pickle.load(open("trained_scaler_popularity.sav", "rb"))
scaler_revenue = pickle.load(open("trained_scaler_revenue.sav", "rb"))
budget = st.number_input("Enter the budget of the movie")
runtime = st.number_input("Enter the duration of the movie in mins")
num_of_production = st.number_input("Enter the number of production companies involved")
description = st.text_input("Enter the description of the movie")
subjective = getSubjectivity(description)
polarity = getPolarity(description)
st.write('Select all the Genres the movie belongs to:')
option_1 = st.checkbox('Comedy')
if option_1:
   comedy = 1
else:
   comedy = 0 
option_12 = st.checkbox('Drama')
if option_12:
   drama = 1
else:
   drama = 0 
option_2 = st.checkbox('Action')
if option_2:
   action = 1
else:
   action = 0 
option_3 = st.checkbox('Adventure')
if option_3:
   adventure = 1
else:
   adventure = 0 
option_4 = st.checkbox('Science Fiction')
if option_4:
   sci_fi = 1
else:
   sci_fi = 0 
option_5 = st.checkbox('Fantasy')
if option_5:
   fantasy = 1
else:
   fantasy = 0 
option_6 = st.checkbox('Animation')
if option_6:
   animation = 1
else:
   animation = 0 
option_7 = st.checkbox('Crime')
if option_7:
   crime = 1
else:
   crime = 0 
option_8 = st.checkbox('Family')
if option_8:
   family = 1
else:
   family = 0 
option_9 = st.checkbox('Thriller')
if option_9:
   thriller = 1
else:
   thriller = 0 
option_10 = st.checkbox('Documentary')
if option_10:
   documentary = 1
else:
   documentary = 0 
option_11 = st.checkbox('Romance')
if option_11:
   romance = 1
else:
   romance = 0 
if(st.button('Predict Popularity and Revenue')):
    input = np.array([budget,runtime,drama, comedy, action, romance ,adventure, sci_fi, fantasy, animation, num_of_production])
    input = scaler.transform(input.reshape(1, -1))
    popular = loaded_model.predict(input)[0]
    if popular >=1:
       st.success("With the following fields the movie would be popular")
    else:
       st.error("With the following fields the movie would not be popular")
    input = np.array([budget,thriller, romance, adventure, crime, family, animation, documentary,num_of_production, polarity, subjective])
    input = scaler_revenue.transform(input.reshape(1, -1))
    popular = loaded_model_revenue.predict(input)[0]
    st.text("Predicted revenue would be {}".format(popular))

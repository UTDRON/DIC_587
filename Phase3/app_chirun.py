
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
import pandas as pd
import calendar

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

    #--------------------
    st.subheader("Trend over Movies Releases of each Month")
    df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])

    # Create a custom categorical data type for months with the correct order
    month_order = list(calendar.month_name)[1:]  # Exclude the empty string at index 0
    df['month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

    # Group data by month and calculate the average number of releases per month
    monthly_release_counts = df.groupby('month_name')['year'].count()   

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_release_counts.index, monthly_release_counts.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Number of Releases')
    plt.title('Monthly Distribution of Releases')
    plt.grid(True)
    st.pyplot(plt.show())



    st.subheader("Trend over Movies Released per Year")
    yearly_movie_count = df['year'].value_counts().sort_index()
    yearly_movie_count = yearly_movie_count[yearly_movie_count > 0]
    # Create a line plot
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_movie_count.index, yearly_movie_count.values, marker='o')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Movies Released')
    plt.title('Number of Movies Released by Year')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.xlim(1901, 2015)
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
# option_1 = st.checkbox('Comedy')
# if option_1:
#    comedy = 1
# else:
#    comedy = 0 
# option_12 = st.checkbox('Drama')
# if option_12:
#    drama = 1
# else:
#    drama = 0 
# option_2 = st.checkbox('Action')
# if option_2:
#    action = 1
# else:
#    action = 0 
# option_3 = st.checkbox('Adventure')
# if option_3:
#    adventure = 1
# else:
#    adventure = 0 
# option_4 = st.checkbox('Science Fiction')
# if option_4:
#    sci_fi = 1
# else:
#    sci_fi = 0 
# option_5 = st.checkbox('Fantasy')
# if option_5:
#    fantasy = 1
# else:
#    fantasy = 0 
# option_6 = st.checkbox('Animation')
# if option_6:
#    animation = 1
# else:
#    animation = 0 
# option_7 = st.checkbox('Crime')
# if option_7:
#    crime = 1
# else:
#    crime = 0 
# option_8 = st.checkbox('Family')
# if option_8:
#    family = 1
# else:
#    family = 0 
# option_9 = st.checkbox('Thriller')
# if option_9:
#    thriller = 1
# else:
#    thriller = 0 
# option_10 = st.checkbox('Documentary')
# if option_10:
#    documentary = 1
# else:
#    documentary = 0 
# option_11 = st.checkbox('Romance')
# if option_11:
#    romance = 1
# else:
#    romance = 0 


# Chnages done by chirun
############
vote_average = st.number_input("Enter the vote average")
vote_count = st.number_input("Enter the vote count")
sentiment = st.number_input("Enter the sentiment score")
subjective = st.number_input("Enter the subjectivity score")
#################
####################
comedy = 1 if st.checkbox('Comedy', key='genre_comedy') else 0
drama = 1 if st.checkbox('Drama', key='genre_drama') else 0
thriller = 1 if st.checkbox('Thriller', key='genre_thriller') else 0
action = 1 if st.checkbox('Action', key='genre_action') else 0
romance = 1 if st.checkbox('Romance', key='genre_romance') else 0
adventure = 1 if st.checkbox('Adventure', key='genre_adventure') else 0
crime = 1 if st.checkbox('Crime', key='genre_crime') else 0
science_fiction = 1 if st.checkbox('Science Fiction', key='genre_science_fiction') else 0
horror = 1 if st.checkbox('Horror', key='genre_horror') else 0
family = 1 if st.checkbox('Family', key='genre_family') else 0
fantasy = 1 if st.checkbox('Fantasy', key='genre_fantasy') else 0
mystery = 1 if st.checkbox('Mystery', key='genre_mystery') else 0
animation = 1 if st.checkbox('Animation', key='genre_animation') else 0
history = 1 if st.checkbox('History', key='genre_history') else 0
music = 1 if st.checkbox('Music', key='genre_music') else 0
war = 1 if st.checkbox('War', key='genre_war') else 0
documentary = 1 if st.checkbox('Documentary', key='genre_documentary') else 0
western = 1 if st.checkbox('Western', key='genre_western') else 0
foreign = 1 if st.checkbox('Foreign', key='genre_foreign') else 0
#######################
if(st.button('Predict Popularity and Revenue')):
    input = np.array([budget,runtime,drama, comedy, action, romance ,adventure, science_fiction, fantasy, animation, num_of_production])
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
##########################
# Combine all genre values into a list
genre_values = [comedy, drama, thriller, action, romance, adventure, crime, science_fiction, horror, family, fantasy, mystery, animation, history, music, war, documentary, western, foreign]

svm_model_popularity = pickle.load(open("saved_models/svm_model.sav", "rb"))

input = np.array([budget, runtime, vote_average, vote_count, *genre_values, num_of_production, sentiment, subjective])
scaler = pickle.load(open("saved_models/scaler.sav", "rb"))

input = scaler.transform(input.reshape(1, -1))

if st.button('Predict Popularity with SVM'):
    prediction = svm_model_popularity.predict(input)
    # Handle the prediction output (adjust based on your labels)
    if prediction[0] == 3:
        st.success("The movie is likely to be very popular.")
    elif prediction[0] == 2:
        st.success("The movie is likely to be moderately popular.")
    else:
        st.error("The movie is unlikely to be popular.")
#------------
#till here






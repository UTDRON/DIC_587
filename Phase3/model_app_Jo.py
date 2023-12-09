import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

#taken from https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524
st.set_option('deprecation.showPyplotGlobalUse', False)
def getSubjectivity(text):

   return TextBlob(str(text)).sentiment.subjectivity
  
 #Create a function to get the polarity
def getPolarity(text):
   return TextBlob(str(text)).sentiment.polarity

# Streamlit app title
st.title("Movie Market Prediction 🎥")
st.write('<p style="font-size:130%">Import Dataset</p>', unsafe_allow_html=True)
dataset = st.file_uploader(label = '')

if dataset:
   st.header("Explorartory Data Analysis for Movie Dataset")

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

   #Removal of rows with zero valued popularity 
   with st.expander("Average Popularity for Different Genre"):
      # st.subheader("Average Popularity for Different Genre")
      df_movie_no_0_popularity = df[(df[['popularity']] != 0).all(axis=1)]
      columns_to_plot = df_movie_no_0_popularity.columns.values.tolist()[24:44]

   #a dictionary to store average popularities for each genre
      average_popularities = {}

   #average popularity for each genre
      for column in columns_to_plot:
         average_popularity = df_movie_no_0_popularity[df_movie_no_0_popularity[column] == 1]['popularity'].mean()
         average_popularities[column] = average_popularity
      
      sorted_popularities = dict(sorted(average_popularities.items(), key=lambda item: item[1]))

      plt.figure(figsize=(9, 6))
      plt.bar(sorted_popularities.keys(), sorted_popularities.values())
      plt.xlabel('Genre')
      plt.ylabel('Average Popularity')
      plt.title('Average Popularity for Different Genre')
      plt.xticks(rotation=45)
      # plt.show()
      st.pyplot(plt.show())

   with st.expander("Average Ratings for Different Genre"):
      # st.subheader("Average Ratings for Different Genre")
      #Removal of rows with zero valued ratings
      df_movie_no_0_ratings = df[(df[['vote_average']] != 0).all(axis=1)]
      average_ratings = {}

      for column in columns_to_plot:
         average_rating = df_movie_no_0_ratings[df_movie_no_0_ratings[column] == 1]['vote_average'].mean()
         average_ratings[column] = average_rating
         
      sorted_ratings = dict(sorted(average_ratings.items(), key=lambda item: item[1]))

      plt.figure(figsize=(9, 6))
      plt.bar(sorted_ratings.keys(), sorted_ratings.values())
      plt.xlabel('Genre')
      plt.ylabel('Average Ratings')
      plt.title('Average Ratings for Different Genre')
      plt.xticks(rotation=45)
      st.pyplot(plt.show())

   with st.expander("Revenue vs. Budget for Movies by Genre"):
      # st.subheader("Revenue vs. Budget for Movies by Genre")
      #Removal of rows with zero valued budgets
      df_movie_no_0_budgets = df[(df[['budget']] != 0).all(axis=1)]
      average_budgets = {}

      for column in columns_to_plot:
         average_budget = df_movie_no_0_budgets[df_movie_no_0_budgets[column] == 1]['budget'].mean()
         average_budgets[column] = average_budget
         
      sorted_budgets = dict(sorted(average_budgets.items(), key=lambda item: item[1]))
         #Removal of rows with zero valued revenue
      df_movie_no_0_revenue = df[(df[['revenue']] != 0).all(axis=1)]
      average_revenues = {}

      for column in columns_to_plot:
         average_revenue = df_movie_no_0_revenue[df_movie_no_0_revenue[column] == 1]['revenue'].mean()
         average_revenues[column] = average_revenue

      sorted_revenues = dict(sorted(average_revenues.items(), key=lambda item: item[1]))
      sorted_revenues.pop('TV Movie')
      genres = list(sorted_revenues.keys())
      sorted_budgets.pop('TV Movie')
      index = np.arange(len(genres))

      bar_width = 0.35

      #grouped bar chart
      plt.figure(figsize=(9, 6))
      plt.bar(index - bar_width/2, list(sorted_revenues.values()), bar_width, label='Revenue', color='b', align='center')
      plt.bar(index + bar_width/2, list(sorted_budgets.values()), bar_width, label='Budget', color='r', align='center')

      plt.xlabel('Genre')
      plt.ylabel('Amount (in billions)')
      plt.title('Revenue vs. Budget for Movies by Genre')
      plt.xticks(index, genres, rotation=45)
      plt.legend()

      plt.tight_layout()
      st.pyplot(plt.show())

   # scatter plot
   #Removal of rows with zero valued runtime and popularity
   with st.expander("Scatter Plot of runtime vs. Popularity"):
      # st.subheader("Scatter Plot of runtime vs. Popularity")
      df_for_plot = df[(df[['runtime','popularity']] != 0).all(axis=1)]
      plt.figure(figsize=(8, 6))
      plt.scatter(df_for_plot['runtime'], df_for_plot['popularity'], color='r', alpha=0.9, s = 10)
      plt.xlabel('Runtime')
      plt.ylabel('popularity')
      plt.title('Scatter Plot of runtime vs. Popularity')
      plt.grid(True)
      # plt.show()
      st.pyplot(plt.show())

   with st.expander("Scatter Plot of No. of Production Companies vs. Budget"):
      # st.subheader("Scatter Plot of No. of Production Companies vs. Budget")
      df_for_plot = df[(df[['num_of_production_companies','budget']] != 0).all(axis=1)]
      plt.figure(figsize=(8, 6))
      plt.scatter(df_for_plot['num_of_production_companies'], df_for_plot['budget'], color='b', alpha=0.99, s = 10)
      plt.xlabel('No. of Production Companies')
      plt.ylabel('Budget')
      plt.title('Scatter Plot of No. of Production Companies vs. Budget')
      plt.grid(True)
      st.pyplot(plt.show())

   # scatter plot
   #Removal of rows with zero valued runtime and revenue
   with st.expander("Scatter Plot of runtime vs. revenue"):
      # st.subheader("Scatter Plot of runtime vs. revenue")
      df_for_plot = df[(df[['runtime','revenue']] != 0).all(axis=1)]
      plt.figure(figsize=(8, 6))
      plt.scatter(df_for_plot['runtime'], df_for_plot['revenue'], color='r', alpha=0.9, s = 10)
      plt.xlabel('Runtime')
      plt.ylabel('Revenue')
      plt.title('Scatter Plot of runtime vs. revenue')
      plt.grid(True)
      st.pyplot(plt.show())

   #box plot for 'budget'
   #Removal of rows with zero valued budget
   with st.expander("Box Plot for Budget"):
      # st.subheader("Box Plot for Budget")
      df_for_box_plot = df[(df[['budget']] != 0).all(axis=1)]
      selected_columns = ['budget']

      plt.figure(figsize=(5, 3))
      df_for_box_plot[selected_columns].boxplot()
      plt.title('Box Plot for Budget')
      plt.ylabel('Value')
      plt.xticks(rotation=45)
      st.pyplot(plt.show())

   #box plot for 'revenue'
   #Removal of rows with zero valued revenue
   with st.expander("Box Plot for Revenue"):
      st.subheader("Box Plot for Revenue")
      df_for_box_plot = df[(df[['revenue']] != 0).all(axis=1)]
      selected_columns = ['revenue']

      plt.figure(figsize=(5, 3))
      df_for_box_plot[selected_columns].boxplot()
      plt.title('Box Plot for revenue')
      plt.ylabel('Value')
      plt.xticks(rotation=45)
      st.pyplot(plt.show())

   #box plot for 'popularity'
   #Removal of rows with zero valued popularity
   with st.expander("Box Plot for Popularity"):
      # st.subheader("Box Plot for Popularity")
      df_for_box_plot = df[(df[['popularity']] != 0).all(axis=1)]
      selected_columns = ['popularity']

      plt.figure(figsize=(5, 3))
      df_for_box_plot[selected_columns].boxplot()
      plt.title('Box Plot for popularity')
      plt.ylabel('Value')
      plt.xticks(rotation=45)
      st.pyplot(plt.show())

   #box plot for 'vote_average'
   #Removal of rows with zero valued vote_average
   with st.expander("Box Plot for Vote Average"):
      # st.subheader("Box Plot for Vote Average")
      df_for_box_plot = df[(df[['vote_average']] != 0).all(axis=1)]
      selected_columns = ['vote_average']

      plt.figure(figsize=(5, 3))
      df_for_box_plot[selected_columns].boxplot()
      plt.title('Box Plot for vote_average')
      plt.ylabel('Value')
      plt.xticks(rotation=45)
      st.pyplot(plt.show())

   #filtering non zero values only
   df_for_heatmap = df[(df[['budget','popularity','revenue', 'runtime','vote_average', 'vote_count']] != 0).all(axis=1)]
   df_for_heatmap = df_for_heatmap.loc[:, ['budget','popularity','revenue', 'runtime','vote_average', 'vote_count','Drama','Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime','Science Fiction', 'Horror', 'Family', 'Fantasy', 'Mystery','Animation', 'History', 'Music', 'War', 'Documentary', 'Western','Foreign', 'num_of_production_companies']]
   sns.heatmap(df_for_heatmap.corr()) 

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
    
    loaded_model_hit_miss =  pickle.load(open("best_xgb_model.sav", "rb"))
    input = np.array([budget,thriller, romance, adventure, crime, family, animation, documentary,num_of_production, polarity, subjective])
    input = scaler_revenue.transform(input.reshape(1, -1))
    popular = loaded_model_hit_miss.predict(input)[0]
    if popular == 2:
       st.success("With the following fields the movie would be hit")
    if popular == 1:
       st.error("With the following fields the movie would be average")
    else:
      st.error("With the following fields the movie would be a miss")

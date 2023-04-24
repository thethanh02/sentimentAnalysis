# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import ktrain
predictor = None
def load_predictor():
	global predictor
	predictor = ktrain.load_predictor('./saved_model')

# Track Utils
from track_utils import *

from googletrans import Translator
translator = Translator()

import sqlite3
def emotiontest_table_exists():
	with sqlite3.connect("database_test10cau.db") as conn:
		c = conn.cursor()
		listOfTables = c.execute(
		"""SELECT name FROM sqlite_master WHERE type='table' AND name='emotionTestTable'; """).fetchall()
		
		if listOfTables == []:
			return False
		return True
	
def load_testdata():
	check = emotiontest_table_exists()
	if not check:
		create_emotiontest_table()
		dataset = pd.read_csv("./app/test10cau.csv")
		for i in range(12):
			raw_text = dataset['Text'][i]
			probability = get_prediction_proba(raw_text)
			add_predictiontest_details(raw_text, predictor.predict(raw_text), float(np.max(probability)), dataset['Emotion'][i])

# Fxn
def predict_emotions(docx):
	results = predictor.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = predictor.predict_proba([docx])
	return results

# emotions_emoji_dict = {'neutral':'😐', 'joy':'😂', 'sadness':'😔', 'fear':'😱', 'surprise':'😮', 'anger':'😠', 'shame':'😥', 'disgust':'🤬'}
emotions_emoji_dict = {'anger':'😠', 'fear':'😱', 'joy':'😂', 'love':'😘', 'sadness':'😔', 'surprise':'😮'}


# Main Application
def main():
	load_predictor()
	load_testdata()
	st.title("Emotion Classifier App")
	menu = ["Home","Monitor","About"]
	choice = st.sidebar.selectbox("Menu",menu)
	create_page_visited_table()
	create_emotionclf_table()
	if choice == "Home":
		add_page_visited_details("Home",datetime.now())
		st.subheader("Home-Emotion In Text")

		with st.form(key='emotion_clf_form'):
			raw_text = st.text_area("Type Here")
			submit_text = st.form_submit_button(label='Submit')

		if submit_text:
			col1,col2  = st.columns(2)
			trans_text = raw_text
			if translator.detect(raw_text).lang == "vi":
				trans_text = translator.translate(raw_text, dest="en").text
			# Apply Fxn Here
			prediction = predict_emotions(trans_text)
			probability = get_prediction_proba(trans_text)
			add_prediction_details(raw_text,prediction,float(np.max(probability)),datetime.now())

			with col1:
				st.success("Original Text")
				st.write(raw_text)

				st.success("Prediction")
				emoji_icon = emotions_emoji_dict[prediction]
				st.write("{}:{}".format(prediction,emoji_icon))
				st.write("Confidence:{}".format(np.max(probability)))



			with col2:
				st.success("Prediction Probability")
				proba_df = pd.DataFrame(probability,columns=predictor.get_classes())
				proba_df_clean = proba_df.T.reset_index()
				proba_df_clean.columns = ["emotions","probability"]

				fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
				st.altair_chart(fig,use_container_width=True)



	elif choice == "Monitor":
		add_page_visited_details("Monitor",datetime.now())
		st.subheader("Monitor App")

		with st.expander("Page Metrics"):
			page_visited_details = pd.DataFrame(view_all_page_visited_details(),columns=['Pagename','Time_of_Visit'])
			st.dataframe(page_visited_details)	

			pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
			c = alt.Chart(pg_count).mark_bar().encode(x='Pagename',y='Counts',color='Pagename')
			st.altair_chart(c,use_container_width=True)	

			p = px.pie(pg_count,values='Counts',names='Pagename')
			st.plotly_chart(p,use_container_width=True)

		with st.expander('Emotion Classifier Metrics'):
			df_emotions = pd.DataFrame(view_all_prediction_details(),columns=['Rawtext','Prediction','Probability','Time_of_Visit'])
			st.dataframe(df_emotions)

			prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
			pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction',y='Counts',color='Prediction')
			st.altair_chart(pc,use_container_width=True)	

	else:
		st.subheader("About")
		add_page_visited_details("About",datetime.now())

		df_emotions = pd.DataFrame(view_all_predictiontest_details(),columns=['RawText','Prediction','Probability','RawEmotion'])
		st.dataframe(df_emotions)
		df_emotions['index'] = df_emotions.index

		chart = alt.Chart(df_emotions).mark_bar(width=20).encode(
			x='index',
			y='Probability',
			color='Prediction'
		)
		st.altair_chart(chart,use_container_width=True)	



if __name__ == '__main__':
	main()
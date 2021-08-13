## Import libraries

import pandas as pd
import pandas_profiling as pp
import streamlit as st

from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

#To visualize the plots
import matplotlib.pyplot as plt
import plotly.express as px

#from backend import initial_cleaning 
from PIL import Image

# This is to supress the warning messages (if any) generated in our code:
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

@st.cache
def load_csv() -> pd.DataFrame:
  """
  function to cache the csv file to avoid memory leak
  """
  csv_to_df = pd.read_csv('BankChurners.csv')
  return csv_to_df

@st.cache
def load_csv2() -> pd.DataFrame:
  """
  function to cache the csv file to avoid memory leak
  """
  csv_to_df = pd.read_csv('final_classification_clients_that_retain.csv')
  return csv_to_df

@st.cache
def load_csv3() -> pd.DataFrame:
  """
  function to cache the csv file to avoid memory leak
  """
  csv_to_df = pd.read_csv('mismatch.csv')
  return csv_to_df

# load data as pandas dataframe
df = load_csv()
df = df.iloc[: , :-2]


st.title("Credit Card Churn Classification")
st.header('Analysis, Model and Evaluation')
st.write('\n')
st.write('by Shilpa Singhal - Data and AI Junior @BeCode')
st.markdown("  >>     ''Big things often have small beginnings ...''")

with st.sidebar:
    st.sidebar.title('Profile Report')
    # visual exploration
    profile = pp.ProfileReport(df, title = 'Data Profile Report',
                         explorative=True, minimal= True)
    st_profile_report(profile)

st.subheader('Important Observation')
st.write('\n')
st.write('Gender Pay Gap is evident')
st.write('The bank has opportunity to tap the potential of high earning female client base.')
image = Image.open('gender_income_marital.png')
st.image(image, caption='Graphical Representation of data by Gender')

st.header('Data Analysis')
col1, col_mid, col2 = st.beta_columns((1, 0.1, 1)) 
with col1:
        st.subheader('')
        image = Image.open('Most common values.png')
        st.image(image, caption='Most common values in the dataset')

with col2:
        st.subheader('')
        st.write('Existing Customers with income more than $120K')
        image = Image.open('120_silver.png')
        st.image(image, caption='')

st.header('Data Modeling Observations')
col1, col_mid, col2 = st.beta_columns((1, 0.1, 1)) 
with col1:
        st.subheader('Hyperparameter tuning')
        #st.write('')
        #st.write('When the curve flattens, no significant impact on accuracy is made by adding number of trees in the forest.')
        image = Image.open('Estimator finding.png')
        st.image(image, caption='RF Accuracy Score vs No. of Estimators')

with col2:
        st.subheader('Model Comparison')
        image = Image.open('Model Comparison.png')
        st.image(image, caption='Comparing different Classication Models')

st.header('Random Forest Model Evaluation')
col1, col_mid, col2 = st.beta_columns((1, 0.1, 1)) 
with col1:
        #st.subheader('Confusion Matrix')
        st.write('How correct are the classifications?')
        #st.write('True Negatives = ')
        # st.write('False Positives = ')
        #st.write('False Negatives = ')
        image = Image.open('test confusion matrix.png')
        st.image(image, caption='Attrited Clients identified as Existing = 24')

with col2:
        st.write('Which attributes are the most important?')
        st.write('\n')
        st.write('\n')
        st.write('\n')
        image = Image.open('imp features.png')
        st.image(image, caption='Transaction Counts and Amount are most important.')

st.header('Classification Insights')
col1, col_mid, col2 = st.beta_columns((1, 0.1, 1)) 
with col1:
        st.write('Customers are categorized based on their propensity to retain.')
        st.write('How much is the Attrition Risk?')
        image = Image.open('Test_Data_Risk_Profile.png')
        st.image(image, caption='Attrited Clients identified as Existing is Costlier Error')

with col2:
        # load data as pandas dataframe
        df = load_csv3()

        st.subheader('')
        # plot the value
        fig = px.bar(df, 
                    y=df.index,
                    x='Number of Customers',
                    width=500, height=400,
                    orientation='h',
                    labels={
                     'index' : "Risk of Attrition",
                     "Number of Customers": "Number of Clients"
                    },     title="Risk profiling for wrongly classified clients - Watch out!")

        fig.update_layout(
                yaxis = dict(
                    tickmode = 'array',
                    tickvals = [0,1,2,3],
                    ticktext = ['Very High', 'High', 'Medium', 'Low']
                )
            )
        
        st.plotly_chart(fig)
 
# load data as pandas dataframe
#df2 = pd.read_csv('final_classification_clients_that_retain.csv',index_col = 0)
df = load_csv2()

# plot the value
fig = px.line(df, 
            x=df.index,
            y='Number of Customers', 
            title="Number of Customers vs. Propensity to retain  - the deal lies in the details")
fig.update_traces(mode="markers+lines", hovertemplate=None)
fig.update_layout(hovermode="x")
st.plotly_chart(fig)

st.write("                                      linkedin - shilpasinghalglo")
st.markdown("  >>     ''Stories are just data with a soul ...Dr.Brene Brown''")



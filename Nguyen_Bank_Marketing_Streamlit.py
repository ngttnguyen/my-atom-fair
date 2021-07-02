import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

def load_data(file_path):
    file_path = "data/bank-additional-full.csv"
    data = pd.read_csv(file_path,sep = ";")
    return data

def convert_str_to_float(mydict):
    for item in mydict.keys():
        mydict[item] = float(mydict[item])
    return mydict
def get_cat_cols_val(data):
    unique_dict= {}
    cat_cols = data.dtypes[data.dtypes == 'object'].index
    for col in cat_cols:
        unique_col = data[col].unique()
        unique_dict[col] = unique_col
    return unique_dict

# Initial setup
st.set_page_config(layout="wide")

#### L O A D  Data
data_file_path = "data/bank-additional-full.csv"
marketing_df = pd.read_csv(data_file_path,sep = ";")

#### L O A D  Model & coefs 
coef_file_path = ("model/model_coefs.json") 
with open(coef_file_path) as infile:
    coef_dict = json.load(infile)
coef_dict = convert_str_to_float(coef_dict)

model_file_path = "model/pkl_model.pkl"
model = pickle.load(open(model_file_path, 'rb'))
    

#### M A I N  Function
def main():
    
    # unique_dict = get_cat_cols_val(marketing_df) #contain unique values of features
    
    st.title('Banking TeleMarketing prediction')
    st.sidebar.markdown('## XGBoost Classifier')
    
  
    st.markdown("Input client's information: ")
    #['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', \
        # 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', \
            # 'cons.conf.idx', 'euribor3m', 'nr.employed']
    cat_cols = marketing_df.dtypes[marketing_df.dtypes == 'object'].index
    for col in cat_cols:

        col_option = marketing_df[col].unique().tolist()
        col_mode = marketing_df[col].mode().tolist()[0]
        col_selected = col_option.index(col_mode)
        col_option_id = st.selectbox('Choose '+col.capitalize()+':',options = col_option, index = col_selected)
            
    # if option_id == learner_option[0]:
    # display_top_learner_report(report_df, nrows = 5)
    # else:
    # display_learner_report(option_id)

  
           
main()

## Run: streamlit run Nguyen_Bank_Marketing_Streamlit.py
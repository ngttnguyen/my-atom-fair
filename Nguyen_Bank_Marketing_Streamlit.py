import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
pd.plotting.register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from xgboost import XGBClassifier
import scikitplot as skplt
import matplotlib.pyplot as plt

def transform_pdays(val):
    transform_dict = {999:'not_previously_contacted',7: 'over_a_week',0:'within_a_week'}
    for key in transform_dict.keys():
        if (val >= key):
            return transform_dict[key]
        
### processing input data
def process_input_client(client_df):
    num_cols = marketing_df.dtypes[marketing_df.dtypes != 'object'].index.tolist()
    cat_cols = [col for col in marketing_df.dtypes[marketing_df.dtypes == 'object'].index.tolist() if col != 'y']

    missing_val = 'unknown'
    un_replaced_lst = ['default']
    for col in cat_cols:
        if (col in un_replaced_lst):
            continue
        replaced_val = marketing_df[col].mode().values.tolist()[0]
        client_df[col] = client_df[col].apply(lambda val: replaced_val if val == missing_val else val)


    for col in num_cols:
        if col == 'pdays':
            client_df[col] = client_df[col].map(transform_pdays)
            cat_cols = cat_cols +['pdays'] 
            continue

        if col == 'campaign':
            replaced_val = 6
        else:
            replaced_val = marketing_df[col].quantile(0.95)   

        client_df[col] = client_df[col].apply(lambda val: replaced_val if val > replaced_val else val)

    ## onehot_encoding:
    cat_cols = [col for col in client_df.dtypes[client_df.dtypes == 'object'].index.tolist()]  
    labelencoder = LabelEncoder()
    for column in cat_cols:
        client_df[column] = labelencoder.fit_transform(client_df[column])
        
    return(client_df)  
 
def get_cat_cols_val(data):
    unique_dict= {}
    cat_cols = data.dtypes[data.dtypes == 'object'].index
    for col in cat_cols:
        unique_col = data[col].unique()
        unique_dict[col] = unique_col
    return unique_dict

# Initial setup
# st.set_page_config(layout="wide")

#### L O A D  Data
data_file_path = "data/bank-additional-full.csv"
marketing_df = pd.read_csv(data_file_path,sep = ";")

model_file_path = "model/pkl_model.pkl"
model = pickle.load(open(model_file_path, 'rb'))



#### M A I N  Function
def main():
    
    client_df_ok = pd.read_csv("client_df.csv", index_col = 'Unnamed: 0')
    
    # client= process_input_client(client_df_ok )
    # print(client)
    # X_client_test = client.drop(['y'], axis = 1) 
    # X_client_test = StandardScaler().fit_transform(X_client_test)
    # print(X_client_test)
    # y_client_pred = model.predict(X_client_test)
    # print(y_client_pred)
    
    # unique_dict = get_cat_cols_val(marketing_df) #contain unique values of features
    
    st.title('Banking TeleMarketing prediction')
    st.sidebar.markdown('## XGBoost Classifier')
    st.markdown("Input client's information: ")

    target = 'y'
    tam = 1
    # num_cols = marketing_df.dtypes[marketing_df.dtypes != 'object'].index.tolist()
    # cat_cols = [col for col in marketing_df.dtypes[marketing_df.dtypes == 'object'].index.tolist() if col != target]
    cols = [col for col in marketing_df.columns.tolist() if col != 'y']
    col_types = [marketing_df[col].dtype.name for col in cols]
    client_df = pd.DataFrame()
    
    for col, col_dtype in zip(cols,col_types):
        if (col_dtype == 'object'):
            col_option_lst = marketing_df[col].unique().tolist()
            col_mode = marketing_df[col].mode().tolist()[0]
            
            col_selected = col_option_lst.index(client_df_ok.iloc[tam][col])
            col_option= st.selectbox('Choose '+col.capitalize()+':',options = col_option_lst, index = col_selected)  
            client_df[col] = [col_option]
        else:
            min_val  = int(marketing_df[col].min())
            max_val  = int(marketing_df[col].max())
            value =  int(client_df_ok.iloc[tam][col])
            col_option = st.slider(str(col), min_value= min_val, max_value= max_val, value=value, step=1)
            
            client_df[col] = [col_option]

    if st.button('Predict'):
            st.write(client_df)
            #client_df = client_df_ok .drop(['y'], axis = 1) 
            client_df = process_input_client(client_df)
            
            X_test = client_df #StandardScaler().fit_transform(client_df)
            
            y_pred = model.predict(X_test)
            st.write('yes' if y_pred == 1 else 'no')
            
           
main()

## Run: streamlit run Nguyen_Bank_Marketing_Streamlit.py
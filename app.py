import streamlit as st 
import numpy as np
import pandas
import sklearn
import matplotlib
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle



model = pickle.load(open('new_model.pkl', 'rb'))

# Load the saved model
#model = pickle.load(open('stacked_model.pkl', 'rb'))

# Function to get user input
def get_user_input():
    # Get input from user for 41 features
    feature_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate']
    
    input_values = []
    for feature in feature_names:
        value = st.number_input(f'Enter value for {feature}', value=0.0, format='%f')
        input_values.append(value)
    
    # Convert input to numpy array
    input_data = np.array(input_values).reshape(1, -1)
    return input_data

# Main function
def main():
    st.title('ICS ATTACK APP')
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">ICS ATTACK PREDICTION APP </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    # Get user input
    input_data = get_user_input()

    if st.button("Predict"): 
        # Make prediction
        prediction = model.predict(input_data)

        print("prediction")
        
        # Display prediction
        st.subheader('Prediction')
        prediction_score = int(prediction[0])
        if prediction_score == 0:
             st.success("This is not an Attack")
             st.write(f"Prediction Score is {prediction_score}")
        else:
            st.error('This is an Attack', icon="🚨")
            st.write(f"Prediction Score is {prediction_score}")



# Run the main function
if __name__ == '__main__':
    main()
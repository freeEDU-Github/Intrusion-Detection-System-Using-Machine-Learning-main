import pandas as pd
import streamlit as st
import numpy as np
import pickle as p
from PIL import Image
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

#Read data for modeling and sample data
df = pd.read_csv("data/final_df.csv")
sample_data = pd.read_csv("data/sample.csv")

#Text
st.title('LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles')
st.markdown("***")

st.header(
          'Proposed ensemble model: Leader Class and Confidence Decision Ensemble (LCCDE)')

st.markdown("LCCDE aims to achieve optimal model performance by identifying the best-performing base ML model with the highest prediction confidence for each class.")

st.markdown("***")

st.subheader("Implementation")
st.markdown("**Dataset**")
st.write("CICIDS2017 dataset, a popular network traffic dataset for intrusion detection problems. Publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html.\n"
         "The dataset will be divided into eighty percent (80%) training sets and twenty percent (20%) validation sets.")

st.markdown("**Data Preprocessing**")
st.write("The CICIDS2017 dataset will undergo to data preprocessing. Since the dataset has 77 features, XGBoost "
         "feature importance has been used to know which features have a larger effect on the model.")

st.markdown("**Machine Learning Algorithms**")
st.markdown(
    """
    - Decision tree (DT)
    - Random forest (RF)
    - Extra trees (ET)
    - XGBoost
    - LightGBM
    - CatBoost
    - Stacking
    - K-means
    """
)

st.write("The best performing algorithm is XGBoost Classifier, which is used in this demonstration.")
image = Image.open('bestmodel.jpg')
st.image(image, caption='Leading Model among the Algorithms')

st.markdown("***")

st.subheader("Prediction")
st.write("You can test the sample data below")
st.markdown(" **Features of our data:** ")
st.text(
        "Bwd Packets Length STD: The standard deviation value of packets in a sub flow in the backward direction\n"
        "Fwd Header Length: The number of packets in a sub flow in the forward direction\n"
        "PSH Flag Count: Instruct the operating system to send (for the sending side) and receive (for the receiving side) the data immediately\n"
        "Total Length of Fwd Packets: The total number of packets in a sub flow in the forward direction\n"
        "Average Packet Size: The average packet size depends on the packet loss rate and the throughput.")
st.dataframe(sample_data)

#Features of our data
bwd_packet_length = st.number_input("Bwd Packet Length Std", step=1e-10, format="%.9f") #Bwd Packet Length Std
fwd_header_length = st.number_input("Fwd Header Length", step=1e-10, format="%.9f") #Fwd Header Length
psh_flag_count = st.number_input("PSH Flag Count", step=1e-10, format="%.9f") #PSH Flag Count
total_length_of_fwd_packets = st.number_input("Total Length of Fwd Packets", step=1e-10, format="%.9f") #Total Length of Fwd Packets
average_packet_size = st.number_input("Average Packet Size", step=1e-10, format="%.9f") #Average Packet Size

features = ['Bwd Packet Length Std', 'Fwd Header Length', 'PSH Flag Count', 'Total Length of Fwd Packets', 'Average Packet Size']

# Labeling X and y features
X = df[features]
y = df['Label']

# Training the data using XGBoost Classifier (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20)
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train,y_train)
predict_val = xgb_clf.predict([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]])
predict_val = float(predict_val)

#Corresponding Attack Types:
if predict_val == 0:
    st.info("Label: Benign")

elif predict_val == 1:
    st.info("Label: Bot")

elif predict_val == 2:
    st.info("Label: BruteForce")

elif predict_val == 3:
    st.info("Label: DoS")

elif predict_val == 4:
    st.info("Label: Infiltration")

elif predict_val == 5:
    st.info("Label: PortScan")

else:
    st.info("Label: WebAttack")
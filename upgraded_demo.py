import pandas as pd
import streamlit as st
import numpy as np
import pickle as p
from PIL import Image
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

#Read data for modeling and sample data
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

image = Image.open('bestmodel.jpg')
st.image(image, caption='Leading Model among the Algorithms', width=None)

st.markdown("***")

st.subheader("Prediction")

st.write("The prediction is done using LCCDE (Leader Class and Confidence Decision Ensemble) which takes into account the predictions and confidence scores of the individual models (LGBM, CatBoost and XGBoost).")

st.write("You can test the sample data below")
st.markdown(" Features of our data: ")
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
average_packet_size = st.number_input("Average Packet Size", step=1e-10, format= "%.9f") #Average Packet Size

features = ['Bwd Packet Length Std', 'Fwd Header Length', 'PSH Flag Count', 'Total Length of Fwd Packets', 'Average Packet Size']

#Labeling X and y features
X = df[features]
y = df['Label']

#Splitting data for training and testing (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20)

#Creating the models to be used in LCCDE
lgbm_clf = LGBMClassifier()
catboost_clf = CatBoostClassifier()
xgb_clf = XGBClassifier()

#Fitting the models with training data
lgbm_clf.fit(X_train, y_train)
catboost_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

#Storing the predictions and confidence scores of each model
lgbm_predictions = lgbm_clf.predict(X_test)
catboost_predictions = catboost_clf.predict(X_test)
xgb_predictions = xgb_clf.predict(X_test)

lgbm_confidence = lgbm_clf.predict_proba(X_test)
catboost_confidence = catboost_clf.predict_proba(X_test)
xgb_confidence = xgb_clf.predict_proba(X_test)

#Storing the predictions and confidence scores of the sample data
sample_predictions = [lgbm_clf.predict([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]]),
catboost_clf.predict([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]]),
xgb_clf.predict([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]])]

sample_confidence = [lgbm_clf.predict_proba([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]]),
catboost_clf.predict_proba([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]]),
xgb_clf.predict_proba([[bwd_packet_length, fwd_header_length, psh_flag_count, total_length_of_fwd_packets, average_packet_size]])]

#Implementing the Leader Class and Confidence Decision Ensemble algorithm
leader_class = max(set([i.item() for i in sample_predictions]), key = [i.item() for i in sample_predictions].count)

if sample_predictions.count(leader_class) > 1:
    leader_confidence = max([sample_confidence[i][0][int(leader_class)] for i in range(len(sample_predictions)) if sample_predictions[i] == leader_class])
else:
    leader_confidence = sample_confidence[sample_predictions.index(leader_class)][0][int(leader_class)]

#Displaying the result
if leader_class == 0:
    st.info("Label: Benign, Confidence: {:.2f}".format(leader_confidence))

elif leader_class == 1:
    st.info("Label: Bot, Confidence: {:.2f}".format(leader_confidence))

elif leader_class == 2:
    st.info("Label: BruteForce, Confidence: {:.2f}".format(leader_confidence))

elif leader_class == 3:
    st.info("Label: DoS, Confidence: {:.2f}".format(leader_confidence))

elif leader_class == 4:
    st.info("Label: Infiltration, Confidence: {:.2f}".format(leader_confidence))

elif leader_class == 5:
    st.info("Label: PortScan, Confidence: {:.2f}".format(leader_confidence))

else:
    st.info("Label: WebAttack, Confidence: {:.2f}".format(leader_confidence))


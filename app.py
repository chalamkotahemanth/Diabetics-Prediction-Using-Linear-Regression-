import pandas as pd
import streamlit as st 
from  sklearn.model_selection import train_test_split
from  sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score, classification_report

df=pd.read_csv(r'C:\Users\hemanth\OneDrive\Desktop\Diabetic Prediction Using Machine Learning\diabetes.csv')
df.isnull().sum()
df.fillna(df.mean(numeric_only =True), inplace =True)
#categorical data
for col in  df.select_dtypes(include="object").columns:

  df[col].fillna(df[col].mode()[0], inplace='True')

#datacleaned
x=df.drop("Outcome", axis=1)
y=df["Outcome"]


scale=StandardScaler()
x_scale=scale.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=30)

model= LogisticRegression()
print(x_train.shape)
print(y_train.shape)
model.fit(x_train,y_train) # model got trained

prediction1=model.predict(x_test)
st.title("🩺 Diabetic Prediction App")

st.write("Enter the health metrics to predict diabetes status:")

Pregnancies=st.number_input("Pregrency",0,20,step=1)
Glucose=st.slider("glucose",1,200,110)
Bloodpressure=st.slider("bloodpressure",1,122,70)
Skinthickness=st.slider("skinthickness",1,99,20)
Insulin=st.slider("insulin",1,846,79)
Bmi=st.slider("bmi",0.0,67.0,32.0)
Diabetespedigreefunction=st.slider("diabetespedigreefunction",0.0,2.5,0.47)
Age=st.slider("age",15,100,33)

if st.button("Predict"):
  user_input=[[Pregnancies,Glucose,Bloodpressure,Skinthickness,Insulin,Bmi,Diabetespedigreefunction,Age]]

  scaler1=scale.transform(user_input)
  result1=model.predict(scaler1)
  if result1[0]==1:
    st.error("The Person be  diabetic")
  else:
    st.success("The person is  non diabetic")

if st.checkbox("Model Evalutation"):
  y_1=model.predict(x_test)
  st.write("Accuracy Score", accuracy_score(y_test,y_1))
  st.text(classification_report(y_test,y_1))






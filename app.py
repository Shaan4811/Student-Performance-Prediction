import streamlit as st
import pandas as pd
import joblib 


model=joblib.load('LinearRegression.pkl')
scaler=joblib.load('scaler.pkl')
encoder=joblib.load('encoder.pkl')


# ----------------------------------------------------------------------
# st.title('Students Performance Predictor')
# st.header('Enter Your Details')
# st.subheader('Details')
# st.write('Here are The Details')
# st.text_input('Enter Your Details')
# st.number_input('Enter the age')
# st.selectbox('choose',('Yes','No'))
# st.radio('Gender',options=['Male','Female'])
# st.button('Submit')

# -------------------------------------------------------------------
# st.title('CALCULATOR')

# num1=st.number_input('Enter The First Number')
# num2=st.number_input('Enter The Second Number')




# choose=st.selectbox('Choose',('Sum','Sub','Div','Mul'))


# if choose=='Sum':
#     if st.button('Submit'):
#         st.success(num1+num2)
# elif choose=='Sub':
#     if st.button('Submit'):
#         st.success(num1-num2)
# elif choose=='Mul':
#     if st.button('Submit'):
#         st.warning(num1*num2)        
# if choose=="Div":
#     if st.button('Submit'):
#         if num2==0:
#             st.error("Cant divisible")
#         else:
#             st.success(num1/num2)        

# ----------------------------------------------------------------

hour_studied=st.number_input('Hours Studied')
prev_score=st.number_input('Previous Scores')
sleep_hours=st.number_input('Sleep Hours')
paper=st.number_input("Sample Question Papers Practiced")
eca=st.selectbox("Extracurricular Activities",('Yes','No'))
eca=encoder.transform([eca])
dataframe=pd.DataFrame({'Hours Studied':hour_studied,'Previous Scores':prev_score,'Sleep Hours':sleep_hours,"Sample Question Papers Practiced":paper,"Extracurricular Activities":eca})
# st.write(dataframe)
scaled_data=scaler.transform(dataframe)
# st.write(scaled_data)
if st.button('predict'):
    prediction=model.predict(scaled_data)[0]
    st.write(prediction)
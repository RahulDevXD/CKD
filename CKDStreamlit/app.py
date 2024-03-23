### Importing Packages
import streamlit as st
import pickle
import pandas as pd
st.set_page_config(layout="wide")
import matplotlib.pyplot as plt
import plotly.express as px


### Loading saved models
loaded_model = pickle.load(open('StreamlitProjects/CKDPrediction/decision_tree_ckd.sav', 'rb'))

###################################################
## Application Header
###################################################
st.markdown('<div style = "background-color: rgb(112, 48, 160); text-align: center; margin-bottom: 5px;"> <h5 style = "color: white; padding: 5px;"> Chronic Kidney Disease Prediction  </h5> </div>', unsafe_allow_html=True)

###################################################
## Defining Side Bar
###################################################
st.sidebar.markdown('''<b>Enter Patient Details:</b>''', unsafe_allow_html=True)

age = st.sidebar.slider("Age", 0, 100, 35)
bp = st.sidebar.radio("Blood pressure", ["Yes", 'No'])
bact = st.sidebar.radio("Bacteria", ["Yes", 'No'])
apt = st.sidebar.radio("Appetite", ["Yes", 'No'])
pedaEdema = st.sidebar.radio("Peda edema", ["Yes", 'No'])

###################################################
## Defining Body
###################################################

col1, col2, col3, col4 = st.columns(4)

highBp = col1.radio("Is Blood Pressure High?", ["Yes", 'No'])
highHemo = col2.radio("Is Hemoglobin Value High?", ["Yes", 'No'])
highNa = col3.radio("High Levels of Sodium?", ["Yes", 'No'])
highDiabe = col4.radio("Is Patient Highly Diabetic?", ["Yes", 'No'])

col1, col2, col3 = st.columns([2, 1, 5])



#### Adding Button
submit = st.button('Get Prediction', key=None, help=None, on_click=None)

if submit:
      ########################################
      #### Data Preparation for Prediction
      ########################################

      ### Creating prediction dataframe
      predDf = pd.DataFrame([{
      'Hemoglobin_High Hemoglobin': 1 if highHemo=='Yes' else 0,
      'Sodium_High Sodium': 1 if highNa=='Yes' else 0,
      'Diabetes mellitus': 1 if highDiabe=='Yes' else 0,
      'Hypertension': 1 if highBp=='Yes' else 0,
      'Age_Updated': age,
      'Peda edema': 1 if pedaEdema=='Yes' else 0,
      'Blood pressure': 1 if bp=='Yes' else 0,
      'Bacteria': 1 if bact=='Yes' else 0,
      'Appetite': 1 if apt=='Yes' else 0,
      }])
      predDf[['Pus cell', 'Pus cell clumps', 'Anemia', 'Coronary artery disease']] = 1

      ## Feature Imp Score
      featureImp = pd.DataFrame({'Feature': ['Hemoglobin_High Hemoglobin', 'Sodium_High Sodium', 'Diabetes mellitus', 'Age_Updated', 'Hypertension', 'Blood pressure', 'Peda edema', 'Pus cell', 'Pus cell clumps', 'Bacteria', 'Coronary artery disease', 'Appetite', 'Anemia'], 
                                 'Importance Score': [0.7267168156133244, 0.10111768773032014, 0.0578765930894093, 0.03746549633540548, 0.0036639956383937744, 0.024753005722123404, 0.015430445125479524, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

      sampDf = predDf.iloc[[0]].T
      sampDf['Feature'] = sampDf.index
      sampDf.columns = ['value', 'Feature']
      finalDf = pd.merge(featureImp, sampDf, how='left', left_on='Feature', right_on='Feature')
      finalDf['Final_Score'] = finalDf['Importance Score']*finalDf['value']
      finalDf = finalDf.sort_values(['Final_Score'], ascending = False).head()
      finalDf = finalDf.sort_values(['Final_Score'])

      ########################################
      #### Getting Model Prediction
      ########################################
      pred = loaded_model.predict_proba(predDf[['Blood pressure', 'Pus cell', 'Pus cell clumps', 'Bacteria',
       'Hypertension', 'Diabetes mellitus', 'Coronary artery disease',
       'Appetite', 'Peda edema', 'Anemia', 'Age_Updated', 'Sodium_High Sodium',
       'Hemoglobin_High Hemoglobin']])[0][0]
      
      predDf = pd.DataFrame({'Class': ['Non CKD', 'CKD'], 'pred': [1-pred,pred]})


      #### Adding pie chart
      fig = px.pie(predDf, values='pred', names='Class', color='Class', color_discrete_map={'CKD':'rgb(253, 66, 61', 'Non CKD':'rgb(146, 208, 80)'}, title = 'CKD Prediction (Probability of CKD vs Not):')
      fig.update_layout(
      autosize=False,
      width=350,
      height=350,
      legend=dict(
            orientation="h",
      yanchor="bottom",
      y=-0.2,
      xanchor="left",
      x=0.01
      )
      )
      col1.plotly_chart(fig)

      #### Adding Imprtant Features
      fig = px.bar(x=finalDf['Final_Score'], y=finalDf['Feature'], color_discrete_sequence=['#F7C0BB'], title = 'Key Factors Driving Patient’s CKD Behaviour:')
      col3.plotly_chart(fig)

      #### Priting message
      if pred <= 0.5:
        st.balloons()
        st.success('The patient is at Low Risk of CKD', icon="✅")
      else:
        st.error('The patient is at High Risk of CKD. Please follow the treatment path accordingly', icon="🚨")

else:
    st.warning('Please click select patient details and click "Get Prediction" ', icon="⚠️")

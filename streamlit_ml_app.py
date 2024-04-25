import joblib
import streamlit as st
import numpy as np

## Load the model
def load_model():
    model_path = "./Model/Trained-Models/RF_loan_model.joblib"
    model = joblib.load(model_path)
    return model

## To parse input to float for model input
def parseData(data):
    if data['Gender'] == "Male":
        data['Gender'] = 1
    else:
        data['Gender'] = 0

    if data['Married'] == "Yes":
        data['Married'] = 1
    else:
        data['Married'] = 0

    if data['Education'] == "Graduate":
        data['Education'] = 0
    else:
        data['Education'] = 1
        
    if data['Self_Employed'] == "Yes":
        data['Self_Employed'] = 1
    else:
        data['Self_Employed'] = 0

    if data['Credit_History'] == "Outstanding Loan":
        data['Credit_History'] = 1
    else:
        data['Credit_History'] = 0   
        
    if data['Property_Area'] == "Rural":
        data['Property_Area'] = 0
    elif data['Property_Area'] == "Semi Urban":
        data['Property_Area'] = 1  
    else:
        data['Property_Area'] = 2

    data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    del data['ApplicantIncome']
    del data['CoapplicantIncome']

    return data

## To generate predictions
def prediction(data):
    data = parseData(data)

    model = load_model()

    pred = model.predict([[data['Gender'],data['Married'],data['Dependents'],data['Education'],data['Self_Employed'],data['LoanAmount'],data['Loan_Amount_Term'],data['Credit_History'],data['Property_Area'],data['Total_Income']]])

    print(pred) ## Print output

    return pred
     

## Front-end for the app
def main():
    ## Headers
    st.title("Welcome to Loan Application")
    st.header("Please enter details to proceed with Loan Application")

    ## Inputs
    Gender = st.selectbox("Gender",("Male","Female"))
    Married = st.selectbox("Married",("Yes","No"))
    Dependents = st.number_input("Number of Dependents")
    Education = st.selectbox("Education",("Graduate","Not Graduate"))
    Self_Employed = st.selectbox("Self Employed",("Yes","No"))
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("LoanAmount")
    Loan_Amount_Term = st.number_input("Loan Amount Term")
    Credit_History = st.selectbox("Credit History",("Outstanding Loan", "No Outstanding Loan"))
    Property_Area = st.selectbox("Property Area",("Rural","Urban","Semi Urban"))

    data = {'Gender':Gender,'Married':Married,'Dependents':Dependents,'Education':Education,'Self_Employed':Self_Employed,'ApplicantIncome':ApplicantIncome,'CoapplicantIncome':CoapplicantIncome,'LoanAmount':LoanAmount,'Loan_Amount_Term':Loan_Amount_Term,'Credit_History':Credit_History,'Property_Area':Property_Area}

    ## Button to predict
    if st.button("Predict"):
        pred = prediction(data)

        ## Prediction and output
        if pred == 1:
            st.success("Your loan Application is Approved")
        elif pred == 0:
            st.error("Your loan Application is Rejected")


## Calling main on executing file
if __name__ == "__main__":
    main()
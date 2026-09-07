"""
Streamlit prediction app.
Loads the registered model from the Hugging Face model hub and serves
predictions for the Wellness Tourism Package.
"""
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "scrfrob/tourism-package-model"
MODEL_FILENAME = "best_tourism_model.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction")
st.write(
    "Predicts whether a customer is likely to purchase the new "
    "Wellness Tourism Package, based on their profile and pitch interaction."
)

st.header("Customer Details")
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox(
    "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Avg. Trips per Year", 0, 25, 2)
Passport = st.selectbox("Holds a Valid Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Children Visiting (below age 5)", 0, 5, 0)
Designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
MonthlyIncome = st.number_input("Monthly Income", 0, 200000, 25000)

st.header("Sales Interaction Details")
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
ProductPitched = st.selectbox(
    "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)
NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 15)

input_data = pd.DataFrame([{
    "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch, "Occupation": Occupation, "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting, "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched, "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting, "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

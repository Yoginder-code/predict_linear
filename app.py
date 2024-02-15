import streamlit as st
import pandas as pd
from joblib import load

def get_user_inputs():
    gre_placeholder = st.empty()
    toefl_placeholder = st.empty()
    university_rating_placeholder = st.empty()
    sop_placeholder = st.empty()
    lor_placeholder = st.empty()
    cgpa_placeholder = st.empty()
    research_placeholder = st.empty()

    with st.form("user_inputs"):
        gre_score = st.text_input("Enter GRE Score (260-340):", key='gre_score_input')
        toefl_score = st.text_input("Enter TOEFL Score (80-120):", key='toefl_score_input')
        university_rating = st.text_input("Enter University Rating (1-5):", key='university_rating_input')
        sop = st.text_input("Enter SOP (1-5):", key='sop_input')
        lor = st.text_input("Enter LOR (1-5):", key='lor_input')
        cgpa = st.text_input("Enter CGPA (1-10):", key='cgpa_input')
        research = st.text_input("Enter Research (0 for No, 1 for Yes):", key='research_input')

        submit_button = st.form_submit_button("Predict Chances of Admit")

        # Validation for out-of-range values
        validate_input_range(gre_score, 260, 340, "GRE Score", gre_placeholder)
        validate_input_range(toefl_score, 80, 120, "TOEFL Score", toefl_placeholder)
        validate_input_range(university_rating, 1, 5, "University Rating", university_rating_placeholder)
        validate_input_range(sop, 1, 5, "SOP", sop_placeholder)
        validate_input_range(lor, 1, 5, "LOR", lor_placeholder)
        validate_input_range(cgpa, 1, 10, "CGPA", cgpa_placeholder)
        validate_input_range(research, 0, 1, "Research", research_placeholder)

    return gre_score, toefl_score, university_rating, sop, lor, cgpa, research, submit_button

def validate_input_range(value, min_value, max_value, prompt, placeholder):
    try:
        if value:
            numeric_value = float(value)
            if min_value <= numeric_value <= max_value:
                return
            else:
                placeholder.warning(f"Please enter a valid numeric value between {min_value} and {max_value} for {prompt}.")
    except ValueError:
        placeholder.warning(f"Invalid input for {prompt}. Please enter a valid numeric value.")

def predict_chances_of_admit(gre_score, toefl_score, university_rating, sop, lor, cgpa, research):
    # Check if all input features are provided
    if any(x is None or x == '' for x in [gre_score, toefl_score, university_rating, sop, lor, cgpa, research]):
        st.warning("Please provide all input features before predicting.")
        return

    # Create a DataFrame with the provided input features
    new_data = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [university_rating],
        'SOP': [sop],
        'LOR ': [lor],
        'CGPA': [cgpa],
        'Research': [research]
    })

    # Load the saved scaler and model
    lasso_scaler = load('lasso_scaler.joblib')
    lasso_model = load('lasso_model.joblib')

    # Standardize the input data
    new_data_scaled = lasso_scaler.transform(new_data)

    # Make predictions using the Lasso model
    chances_of_admit = lasso_model.predict(new_data_scaled)

    predicted_chances_of_admit = chances_of_admit[0].round(2) * 100

    # Display the prediction result
    st.success(f"Predicted Chances of Admit: {predicted_chances_of_admit}%")

    # Notify the user to refresh the page
    st.warning("Please refresh the page to re-run the app with new inputs.")

def main():
    st.title("Admission Chances Predictor")

    gre_score, toefl_score, university_rating, sop, lor, cgpa, research, submit_button = get_user_inputs()

    if submit_button:
        predict_chances_of_admit(gre_score, toefl_score, university_rating, sop, lor, cgpa, research)

if __name__ == '__main__':
    main()

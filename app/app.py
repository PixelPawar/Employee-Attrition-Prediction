import streamlit as st
import pandas as pd
import joblib

model = joblib.load("../models/employee_attrition_model.pkl");

col1 , col2 = st.columns(2);

with col1:
    age = st.number_input("Age", 18, 60, 30)
    business_travel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )
    daily_rate = st.number_input("Daily Rate", 100, 2000, 800)
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    distance = st.number_input("Distance From Home", 1, 30, 5)
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hourly_rate = st.number_input("Hourly Rate", 30, 100, 60)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

with col2:
    monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
    monthly_rate = st.number_input("Monthly Rate", 1000, 30000, 15000)
    num_companies = st.number_input("Number of Companies Worked", 0, 10, 2)
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
    performance = st.selectbox("Performance Rating", [3, 4])
    relationship = st.slider("Relationship Satisfaction", 1, 4, 3)
    stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    total_working_years = st.number_input("Total Working Years", 0, 40, 10)
    training = st.number_input("Training Times Last Year", 0, 10, 2)
    work_life = st.slider("Work Life Balance", 1, 4, 3)
    years_company = st.number_input("Years At Company", 0, 40, 5)
    years_role = st.number_input("Years In Current Role", 0, 20, 3)
    years_promotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
    years_manager = st.number_input("Years With Current Manager", 0, 20, 3)

if st.button("Predict Attrition"):
    input_data = pd.DataFrame({
    "Age": [age],
    "BusinessTravel": [business_travel],
    "DailyRate": [daily_rate],
    "Department": [department],
    "DistanceFromHome": [distance],
    "Education": [education],
    "EducationField": [education_field],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "Gender": [gender],
    "HourlyRate": [hourly_rate],
    "JobInvolvement": [job_involvement],
    "JobLevel": [job_level],
    "JobRole": [job_role],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status],
    "MonthlyIncome": [monthly_income],
    "MonthlyRate": [monthly_rate],
    "NumCompaniesWorked": [num_companies],
    "OverTime": [overtime],
    "PercentSalaryHike": [salary_hike],
    "PerformanceRating": [performance],
    "RelationshipSatisfaction": [relationship],
    "StockOptionLevel": [stock_option],
    "TotalWorkingYears": [total_working_years],
    "TrainingTimesLastYear": [training],
    "WorkLifeBalance": [work_life],
    "YearsAtCompany": [years_company],
    "YearsInCurrentRole": [years_role],
    "YearsSinceLastPromotion": [years_promotion],
    "YearsWithCurrManager": [years_manager]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]



    with st.sidebar:
        st.title("Employee Attrition")

        st.info(
            """
            This application predicts whether an employee
            is likely to leave the company.

            
            """
        )

        st.markdown("---")

        stay_prob = probability[0]
        leave_prob = probability[1]

        st.subheader("Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Stay", f"{stay_prob*100:.2f}%")
            st.progress(stay_prob)

        with col2:
            st.metric("Leave", f"{leave_prob*100:.2f}%")
            st.progress(leave_prob)

        if prediction == 1:

            st.error(" High Attrition Risk")

            st.write(
                """
                This employee has a high probability
                of leaving the company.

                HR should consider intervention.
                """
            )

        else:

            st.success(" Low Attrition Risk")

            st.write(
                """
                Employee is likely to remain
                in the company.
                """
            )

    st.subheader("Employee Summary")

    st.dataframe(input_data)

    confidence = max(probability)

    st.metric(
        "Model Confidence",
        f"{confidence*100:.2f}%"
    )

    st.markdown("---")

    st.caption(
        "Employee Attrition Prediction using Machine Learning"
    )
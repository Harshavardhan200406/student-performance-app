import streamlit as st
import joblib
import pandas as pd
import time
import random

# Load the trained model
try:
    model = joblib.load('student_performance_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run 'train_model.py' first.")
    st.stop()

# --- 1. Chatbot Logic (Rule-Based) ---
def get_bot_response(user_input):
    user_input = user_input.lower()
    
    # Simple Keyword Matching
    if "stress" in user_input or "worried" in user_input:
        return "I understand. Exams can be stressful. Try the '5-4-3-2-1' grounding technique to calm down."
    elif "sleep" in user_input or "tired" in user_input:
        return "Sleep is crucial for memory. Try to get at least 7 hours tonight. Avoid screens 1 hour before bed."
    elif "grades" in user_input or "marks" in user_input or "fail" in user_input:
        return "Don't panic about grades. Focus on one subject at a time. Have you tried the Pomodoro technique?"
    elif "hello" in user_input or "hi" in user_input:
        return "Hello there! I am your Student Support Bot. How are you feeling today?"
    elif "sad" in user_input or "depressed" in user_input:
        return "I'm sorry you're feeling this way. Please talk to a friend or campus counselor. You are not alone. 💙"
    else:
        return "I'm listening. Tell me more about your study habits or daily routine."

# --- 2. Main App UI ---
def main():
    st.set_page_config(page_title="Student Mental Health System", layout="wide")
    
    # Sidebar for Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Prediction System", "Counselor Chatbot"])

    # --- PAGE 1: Prediction System ---
    if app_mode == "Prediction System":
        st.title("雌 Student Performance Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Academic Data")
            study_hours = st.slider("Daily Study Hours", 0, 15, 4)
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            marks = st.number_input("Average Marks", 0, 100, 60)
            backlogs = st.number_input("Number of Backlogs", 0, 10, 0)

        with col2:
            st.subheader("Behavioral Data")
            sleep_hours = st.slider("Daily Sleep Hours", 0, 12, 7)
            screen_time = st.slider("Daily Screen Time (Hrs)", 0, 15, 5)

        if st.button("Predict Performance"):
            input_data = pd.DataFrame([[study_hours, attendance, marks, backlogs, sleep_hours, screen_time]],
                                      columns=['StudyHours', 'Attendance', 'Marks', 'Backlogs', 'SleepHours', 'ScreenTime'])
            
            with st.spinner('Analyzing...'):
                time.sleep(1) 
                prediction = model.predict(input_data)[0]

            st.divider()
            st.subheader(f"Predicted Performance: {prediction}")
            
            if prediction == 'Excellent':
                st.success("Great job! Keep maintaining this balance.")
            elif prediction == 'Good':
                st.info("Good work! A little more focus on weak areas will help.")
            elif prediction == 'Average':
                st.warning("Warning: Focus on attendance and reducing screen time.")
            else:
                st.error("Alert: Please reach out to a mentor. Prioritize sleep and mental health.")

    # --- PAGE 2: Counselor Chatbot ---
    elif app_mode == "Counselor Chatbot":
        st.title("玄 AI Student Counselor")
        st.write("Chat with me about stress, sleep, or study tips!")

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input Field
        if prompt := st.chat_input("Type here... (e.g., 'I feel stressed')"):
            
            # 1. Show User Message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 2. Generate Bot Response
            response = get_bot_response(prompt)

            # 3. Show Bot Message
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
import streamlit as st
import joblib
import pandas as pd
import time
import google.generativeai as genai

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Student Mental Health System", layout="wide")

# !!! IMPORTANT: REPLACE THIS WITH YOUR ACTUAL GOOGLE API KEY !!!
GOOG_API_KEY = "AIzaSyBUeUTzbAnJE5eDquWdSpzUjMvP0phRdng"

# Configure the AI Model
try:
    genai.configure(api_key=GOOG_API_KEY)
    ai_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Error configuring AI: {e}")

# Load the Prediction Model
try:
    prediction_model = joblib.load('student_performance_model.pkl')
except FileNotFoundError:
    # Create a dummy model if file is missing (prevents app crash during testing)
    prediction_model = None 

# --- 2. AI Chatbot Logic ---
def get_gemini_response(user_input, chat_history):
    """
    Sends the user input and conversation history to Gemini AI
    to get a smart, context-aware response.
    """
    if "PASTE_YOUR_API_KEY_HERE" in GOOG_API_KEY:
        return "⚠️ Please paste your Google API Key in the code (Line 12) to use the chatbot."

    try:
        # Define the AI's persona
        system_instruction = (
            "You are a supportive, empathetic Student Counselor Bot. "
            "Your goal is to help students with stress, study habits, and sleep schedules. "
            "Keep answers concise (under 3 sentences) and friendly. "
            "CRITICAL SAFETY RULE: If a student mentions self-harm, suicide, or severe depression, "
            "do not try to treat them. Instead, strictly advise them to contact a professional or a helpline immediately."
        )

        # Build history for Gemini (Convert Streamlit format to Gemini format)
        gemini_history = []
        
        # Add system instruction as the first hidden context
        gemini_history.append({'role': 'user', 'parts': [system_instruction]})
        gemini_history.append({'role': 'model', 'parts': ["Understood. I will act as a supportive counselor."]})

        # Append actual chat history
        for msg in chat_history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({'role': role, 'parts': [msg["content"]]})

        # Generate response
        chat = ai_model.start_chat(history=gemini_history)
        response = chat.send_message(user_input)
        return response.text

    except Exception as e:
        return f"I'm having trouble connecting to the server. Error: {str(e)}"

# --- 3. Main App UI ---
def main():
    
    # Sidebar for Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Prediction System", "Counselor Chatbot"])

    # --- PAGE 1: Prediction System ---
    if app_mode == "Prediction System":
        st.title("📊 Student Performance Prediction")
        
        if prediction_model is None:
            st.warning("⚠️ 'student_performance_model.pkl' not found. Please run 'train_model.py' first.")
            st.stop()

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
            
            with st.spinner('Analyzing patterns...'):
                time.sleep(1) 
                prediction = prediction_model.predict(input_data)[0]

            st.divider()
            st.subheader(f"Predicted Performance: {prediction}")
            
            if prediction == 'Excellent':
                st.success("🌟 Great job! Keep maintaining this balance.")
            elif prediction == 'Good':
                st.info("👍 Good work! A little more focus on weak areas will help.")
            elif prediction == 'Average':
                st.warning("⚠️ Warning: Focus on attendance and reducing screen time.")
            else:
                st.error("🚨 Alert: Please reach out to a mentor. Prioritize sleep and mental health.")

    # --- PAGE 2: Counselor Chatbot ---
    elif app_mode == "Counselor Chatbot":
        st.title("🤖 AI Student Counselor")
        st.markdown("I am powered by **Gemini AI**. I can help you plan studies, manage stress, or just chat!")

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input Field
        if prompt := st.chat_input("Tell me what's on your mind..."):
            
            # 1. Show User Message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # 2. Generate Bot Response with a spinner
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # We pass the history excluding the latest message we just appended (to avoid duplication logic)
                    response_text = get_gemini_response(prompt, st.session_state.messages[:-1])
                    st.markdown(response_text)
            
            # 3. Save Bot Message to History
            st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == '__main__':
    main()

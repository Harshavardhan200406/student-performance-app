import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# 1. Generate Data (3000 students)
n_samples = 3000 

study_hours = np.random.randint(1, 10, n_samples)
attendance = np.random.randint(50, 100, n_samples)
marks = np.random.randint(40, 100, n_samples)
backlogs = np.random.randint(0, 4, n_samples) # 0 to 3 backlogs
sleep_hours = np.random.randint(4, 10, n_samples)
screen_time = np.random.randint(1, 10, n_samples)

df = pd.DataFrame({
    'StudyHours': study_hours,
    'Attendance': attendance,
    'Marks': marks,
    'Backlogs': backlogs,
    'SleepHours': sleep_hours,
    'ScreenTime': screen_time
})

# 2. DEFINE STRICT LOGIC
def categorize_performance(row):
    # Calculate base score
    score = (row['Marks'] * 0.5) + (row['Attendance'] * 0.3) + (row['StudyHours'] * 5)
    
    # --- THE FIX: STRICT BACKLOG RULE ---
    # If you have ANY backlog, you cannot be 'Excellent' or 'Good'.
    if row['Backlogs'] >= 1:
        # Even if marks are high, max rating is 'Average'
        if score >= 50:
            return 'Average'
        else:
            return 'Poor'
    
    # Standard rules for students with NO backlogs
    if score >= 85: return 'Excellent'
    elif score >= 65: return 'Good'
    elif score >= 45: return 'Average'
    else: return 'Poor'

df['Performance'] = df.apply(categorize_performance, axis=1)

# Save
df.to_csv('student_data.csv', index=False)
print("Success! Generated data with STRICT backlog rules.")
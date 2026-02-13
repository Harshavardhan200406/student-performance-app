import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the new large dataset
data = pd.read_csv('student_data.csv')

X = data[['StudyHours', 'Attendance', 'Marks', 'Backlogs', 'SleepHours', 'ScreenTime']]
y = data['Performance']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a More Powerful Model
# n_estimators=200 makes it "think" harder.
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--------------------------------")
print(f"NEW Model Accuracy: {accuracy * 100:.2f}%")
print(f"--------------------------------")
print(classification_report(y_test, y_pred))

# 5. Save
joblib.dump(model, 'student_performance_model.pkl')
print("High-accuracy model saved!")
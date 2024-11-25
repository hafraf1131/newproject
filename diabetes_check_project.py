import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load and train the model (This should ideally be done only once)
# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns)

# Split the data into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data (Standardize features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 2: Create the Tkinter GUI for the application

# Global variable to store the selected gender
selected_gender = None

def predict_diabetes():
    try:
        # Collect user input from the entry fields
        if selected_gender == 'male':
            # Skip pregnancies for male
            pregnancies = 0  # Set a default value or skip
        else:
            pregnancies = int(entry_pregnancies.get())
        
        glucose = float(entry_glucose.get())
        blood_pressure = float(entry_blood_pressure.get())
        skin_thickness = float(entry_skin_thickness.get())
        insulin = float(entry_insulin.get())
        bmi = float(entry_bmi.get())
        diabetes_pedigree = float(entry_diabetes_pedigree.get())
        age = int(entry_age.get())

        # Prepare the input for prediction (standardize it as the model was trained on standardized data)
        input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

        # Predict whether the person has diabetes (1) or not (0)
        prediction = model.predict(input_data)

        # Show result to the user
        if prediction == 1:
            messagebox.showinfo("Prediction Result", "The person is likely to have diabetes.")
        else:
            messagebox.showinfo("Prediction Result", "The person is likely not to have diabetes.")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid values for all fields.")

def show_diabetes_questions(gender):
    global selected_gender
    selected_gender = gender  # Store the selected gender
    
    # Hide the gender selection screen
    frame_gender.pack_forget()

    # Show the input form for diabetes prediction
    frame_input.pack()

    # Conditionally remove the "Pregnancies" field for males
    if selected_gender == 'male':
        label_pregnancies.pack_forget()
        entry_pregnancies.pack_forget()
    else:
        label_pregnancies.pack()
        entry_pregnancies.pack()

# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction")

# Frame for Gender Selection
frame_gender = tk.Frame(root)

label_gender = tk.Label(frame_gender, text="Please select your gender:")
label_gender.pack()

btn_male = tk.Button(frame_gender, text="Male", width=20, command=lambda: show_diabetes_questions('male'))
btn_male.pack(pady=5)

btn_female = tk.Button(frame_gender, text="Female", width=20, command=lambda: show_diabetes_questions('female'))
btn_female.pack(pady=5)

frame_gender.pack(pady=20)

# Frame for input questions (after selecting gender)
frame_input = tk.Frame(root)

label_pregnancies = tk.Label(frame_input, text="Number of Pregnancies:")
label_pregnancies.pack()
entry_pregnancies = tk.Entry(frame_input)
entry_pregnancies.pack()

label_glucose = tk.Label(frame_input, text="Glucose Level:")
label_glucose.pack()
entry_glucose = tk.Entry(frame_input)
entry_glucose.pack()

label_blood_pressure = tk.Label(frame_input, text="Blood Pressure:")
label_blood_pressure.pack()
entry_blood_pressure = tk.Entry(frame_input)
entry_blood_pressure.pack()

label_skin_thickness = tk.Label(frame_input, text="Skin Thickness:")
label_skin_thickness.pack()
entry_skin_thickness = tk.Entry(frame_input)
entry_skin_thickness.pack()

label_insulin = tk.Label(frame_input, text="Insulin Level:")
label_insulin.pack()
entry_insulin = tk.Entry(frame_input)
entry_insulin.pack()

label_bmi = tk.Label(frame_input, text="BMI:")
label_bmi.pack()
entry_bmi = tk.Entry(frame_input)
entry_bmi.pack()

label_diabetes_pedigree = tk.Label(frame_input, text="Diabetes Pedigree Function:")
label_diabetes_pedigree.pack()
entry_diabetes_pedigree = tk.Entry(frame_input)
entry_diabetes_pedigree.pack()

label_age = tk.Label(frame_input, text="Age:")
label_age.pack()
entry_age = tk.Entry(frame_input)
entry_age.pack()

# Submit Button
btn_submit = tk.Button(frame_input, text="Submit", width=20, command=predict_diabetes)
btn_submit.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()

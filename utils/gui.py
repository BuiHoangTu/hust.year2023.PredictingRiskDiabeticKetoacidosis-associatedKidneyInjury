import tkinter as tk
from tkinter import Label, ttk

import pandas as pd
import requests

prediction_label: None | Label = None

# Function to submit data and display it in the table
def submit_data():
    stay_id = stay_id_entry.get()
    measurement = measurement_entry.get()
    value = value_entry.get()
    time = time_entry.get()

    # check stay_id empty or onlt white space
    if not stay_id.strip():
        return
    
    data = {
        "stay_id": stay_id,
        "measurement": measurement,
        "value": value,
        "time": time,
    }
    
    try:
        res = requests.post("http://localhost:5000/", json=data)
        
        if res.status_code == 201:
            table.delete(*table.get_children())
            
            # server return json of key, value as string
            for key, value in res.json().items():
                table.insert("", "end", values=(key, value))
            
            if prediction_label:
                prediction_label.grid_remove()
            table.grid(row=table_row, column=0, columnspan=2, pady=10)
    except requests.exceptions.ConnectionError:
        # do nothing 
        pass


# Function to handle predictions
def predict():
    stay_id = stay_id_entry.get()
    
    # check stay_id empty or onlt white space
    if not stay_id.strip():
        return
    
    try:
        res = requests.get(f"http://localhost:5000/{stay_id}")
        
        if res.status_code == 200:
            prediction = res.json()["prediction"]
            explanation = res.json().get("explanation")
            
            res_label.config(text=f"Result: {prediction}")

            if explanation:            
                # hide the table and display the explanation there (explaination is encoded png) 
                table.grid_remove()

                # display the explanation
                img = tk.PhotoImage(data=explanation)
                explanation_label = tk.Label(root, image=img)
                # explanation_label.image = img
                explanation_label.grid(row=table_row, column=0, columnspan=2, pady=10)
    except requests.exceptions.ConnectionError:
        # do nothing 
        pass        


# Create the main window
root = tk.Tk()
root.title("Data Entry and Prediction")

# Create input fields and labels
row = 0

stay_id_label = tk.Label(root, text="Stay ID:")
stay_id_label.grid(row=row, column=0, padx=10, pady=10)
stay_id_entry = tk.Entry(root)
stay_id_entry.grid(row=row, column=1, padx=10, pady=10)
row += 1

measurement_label = tk.Label(root, text="Measurement:")
measurement_label.grid(row=row, column=0, padx=10, pady=10)
measurement_entry = tk.Entry(root)
measurement_entry.grid(row=row, column=1, padx=10, pady=10)
row += 1

value_label = tk.Label(root, text="Value:")
value_label.grid(row=row, column=0, padx=10, pady=10)
value_entry = tk.Entry(root)
value_entry.grid(row=row, column=1, padx=10, pady=10)
row += 1

time_label = tk.Label(root, text="Time:")
time_label.grid(row=row, column=0, padx=10, pady=10)
time_entry = tk.Entry(root)
time_entry.grid(row=row, column=1, padx=10, pady=10)
row += 1

# Create Submit button
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# Create table to display data
table_row = row
columns = ("measurement", "value")
table = ttk.Treeview(root, columns=columns, show="headings")
table.heading("measurement", text="Measurement")
table.heading("value", text="Value")
table.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# result 
res_label = tk.Label(root, text="Result:")
res_label.grid(row=row, column=0, padx=10, pady=10)
res_entry = tk.Entry(root)

# Create Predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# Run the application
root.mainloop()

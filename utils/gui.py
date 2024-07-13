import tkinter as tk
from tkinter import ttk


# Function to submit data and display it in the table
def submit_data():
    stay_id = stay_id_entry.get()
    measurement = measurement_entry.get()

    if stay_id and measurement:
        table.insert("", "end", values=(stay_id, measurement))
        stay_id_entry.delete(0, tk.END)
        measurement_entry.delete(0, tk.END)


# Function to handle predictions
def predict():
    # Dummy prediction function
    print("Prediction logic goes here")


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

# Create Submit button
submit_button = tk.Button(root, text="Submit", command=submit_data)
submit_button.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# Create table to display data
columns = ("measurement", "value")
table = ttk.Treeview(root, columns=columns, show="headings")
table.heading("measurement", text="Measurement")
table.heading("value", text="Value")
table.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# Create Predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=row, column=0, columnspan=2, pady=10)
row += 1

# Run the application
root.mainloop()

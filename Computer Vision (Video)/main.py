import tkinter as tk
import threading
import subprocess
def run_script(script_name):
    root.destroy() 
    threading.Thread(target=subprocess.run, args=(["python", script_name],)).start()
root = tk.Tk()
root.title("Activity Menu")
root.geometry("300x200")
backup_button = tk.Button(root, text="Run Back Up", command=lambda: run_script("back_up.py"), width=20, height=2)
backup_button.pack(pady=10)

face_rec_button = tk.Button(root, text="Run Face Recognition", command=lambda: run_script("face_recognition.py"), width=20, height=2)
face_rec_button.pack(pady=10)

root.mainloop()
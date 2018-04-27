from tkinter import *
from ui.gui_utils import rec, center_window


def on_record():
    global action
    action.set("Recording...")
    rec()
    action.set("Enter answer...")

root = Tk()
center_window(root, 400, 400)
root.wm_title('Voice Emotion Recognizer')

theLabel = Label(root, text="How are you Feeling today?", font=("Helvetical", 18), bg="blue")
theLabel.pack(fill=X)
topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side=TOP)

button1 = Button(topFrame, text="RECORD", fg="red", command=on_record)
button3 = Button(topFrame, text="ANALYSIS", fg="green")
button4 = Button(bottomFrame, text="QUIT/CANCEL", fg="black", command=root.destroy)
button1.pack(side=LEFT)

button3.pack(side=LEFT)
button4.pack(side=RIGHT)

action = StringVar()
action.set(value="Enter answer")
Label(root, text=action.get(), textvariable=action, font=("helvetical", 12)).pack()

root.mainloop()

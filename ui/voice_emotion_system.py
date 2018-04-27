from tkinter import *
from tkinter.messagebox import showinfo
from utils.model_utils import Predictor
from ui.gui_utils import RecorderThread, PredictionThread


class Application:

    window_title = "Voice Emotion Recognizer"
    main_label = "How are you Feeling today?"
    record_label = "Record"
    analyze_label = "Analyze"
    quit_label = "Quit"

    def __init__(self):
        self.root = Tk()
        self.build_strings()
        self.build_predictor()

        self.root.wm_title(self.window_title)

        label_1 = Label(text=self.main_label, fg="white", font=("Helvetical", 16), bg="blue")
        label_1.grid(row=0, column=0, columnspan=10, pady=(0, 30))

        self.button_1 = Button(self.root, textvariable=self.record_label, fg="red", height=2, command=self.on_record)
        self.button_1.grid(row=2, column=1, columnspan=3, pady=(0, 10))

        self.button_2 = Button(self.root, textvariable=self.analyze_label, fg="black", height=2, state=DISABLED,
                               command=self.on_analyze)
        self.button_2.grid(row=2, column=6, columnspan=3, pady=(0, 10))

        self.button_3 = Button(self.root, text=self.quit_label, fg="black", height=2, command=self.root.destroy)
        self.button_3.grid(row=3, column=4, columnspan=2, pady=(0, 10))

    def on_analyze(self):
        self.analyze_label.set("Analyzing...")

        pred = PredictionThread(predictor=self.predictor, file_path="./output.wav", callback_fn=self.after_analyze)
        pred.start()

    def after_analyze(self, emotional_state):
        self.analyze_label.set("Analyze")
        showinfo("Emotional State", emotional_state + "    ")

    def on_record(self):
        self.record_label.set("Recording...")

        recorder = RecorderThread(self.after_record)
        recorder.start()

    def after_record(self):
        self.record_label.set("Re-Record")
        self.button_2['state'] = "normal"

    def build_strings(self):
        self.window_title = "Voice Emotion Recognizer"
        self.main_label = "How are you Feeling today?"
        self.quit_label = "Quit"

        self.record_label = StringVar()
        self.analyze_label = StringVar()

        self.record_label.set("Record")
        self.analyze_label.set("Analyze")

    def build_predictor(self):
        self.predictor = Predictor(config_path="/home/hugolucas/PycharmProjects/sound/configs/config_1.json",
                                   model_state="/home/hugolucas/PycharmProjects/sound/models/final_model.pt")

app = Application()
app.root.mainloop()

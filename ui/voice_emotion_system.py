from tkinter import *
from tkinter.messagebox import showinfo

from utils.gui_utils import RecorderThread, PredictionThread
from utils.model_utils import Predictor


class Application:
    """
    A simple GUI that allows the User to record themselves and pass the audio data to a trained model.
    """

    def __init__(self):
        """
        Creates a Predictor object in order to classify audio at a later time and builds the GUI once that is complete.
        """
        self.predictor = Predictor(config_path="/home/hugo/PycharmProjects/emotion_classification/configs/config_1.json",
                                   model_state="/home/hugo/PycharmProjects/emotion_classification/final_model.pt")
        self.root = Tk()
        self.build_strings()
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
        """
        Triggers the PredictorThread in order to classify the audio data recorded by the User.

        :return: None
        """
        self.analyze_label.set("Analyzing...")

        pred = PredictionThread(predictor=self.predictor,
                                file_path="/home/hugo/PycharmProjects/emotion_classification/ui/output.wav",
                                callback_fn=self.after_analyze)
        pred.start()

    def after_analyze(self, emotional_state):
        """
        Updates the GUI once a prediction has been made.

        :param emotional_state:     the string output of the Predictor.make_single_prediction method
        :return: None
        """
        self.analyze_label.set("Analyze")
        showinfo("Emotional State", emotional_state + "    ")

    def on_record(self):
        """
        Triggers the RecorderThread in order to record the User's voice

        :return: None
        """
        self.record_label.set("Recording...")

        recorder = RecorderThread(self.after_record)
        recorder.start()

    def after_record(self):
        """
        Updates the GUI once the recording is complete.

        :return: None
        """
        self.record_label.set("Re-Record")
        self.button_2['state'] = "normal"

    def build_strings(self):
        """
        Declares the String labels for many of the prompts and buttons used in this GUI.

        :return: None
        """
        self.window_title = "Voice Emotion Recognizer"
        self.main_label = "How are you Feeling today?"
        self.quit_label = "Quit"

        self.record_label = StringVar()
        self.analyze_label = StringVar()

        self.record_label.set("Record")
        self.analyze_label.set("Analyze")

app = Application()
app.root.mainloop()

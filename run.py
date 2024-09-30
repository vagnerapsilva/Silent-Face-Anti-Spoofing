import os.path
import datetime
import pickle

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import util
from test import test


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.result_label = tk.Label(self.main_window, text="", font=("Helvetica", 16), fg="blue")
        self.result_label.place(x=750, y=200)

        self.add_webcam(self.webcam_label)

        self.db_dir = './Silent_Face_Anti_Spoofing/db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './Silent_Face_Anti_Spoofing/log.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to capture frame. Retrying...")
            self._label.after(20, self.process_webcam)  # Retry after a short delay
            return  # Exit if frame capture fails

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        # Run face recognition every 500ms
        self.recognize_face()
        self._label.after(500, self.process_webcam)

    def recognize_face(self):
        # Detect all face locations in the frame
        face_locations = face_recognition.face_locations(self.most_recent_capture_arr)
        face_encodings = face_recognition.face_encodings(self.most_recent_capture_arr, face_locations)

        if len(face_encodings) == 0:
            self.result_label.config(text="No faces detected", fg="red")
            return

        # Create a copy of the original frame to draw boxes
        frame_with_boxes = self.most_recent_capture_arr.copy()

        names = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Recognize face by comparing encodings
            name = util.recognize_face_encoding(face_encoding, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                name = "Unknown"
            
            names.append(name)

            # Draw a box around the face
            cv2.rectangle(frame_with_boxes, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the name (or "Unknown") below the face
            cv2.rectangle(frame_with_boxes, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_with_boxes, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Convert the frame with boxes to RGB for displaying in Tkinter
        img_with_boxes_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        imgtk_with_boxes = ImageTk.PhotoImage(image=Image.fromarray(img_with_boxes_rgb))

        # Update the webcam label to show the frame with boxes and names
        self._label.imgtk = imgtk_with_boxes
        self._label.configure(image=imgtk_with_boxes)

        # Display all recognized names in the result label
        self.result_label.config(text=f"Recognized: {', '.join(names)}", fg="green" if "Unknown" not in names else "red")

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user_window.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        encodings = face_recognition.face_encodings(self.register_new_user_capture)
        if len(encodings) == 0:
            util.msg_box('Error', 'No face detected. Please capture a new image with a visible face.')
            return  # Exit if no encodings found

        embeddings = encodings[0]

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')
        self.register_new_user_window.destroy()

if __name__ == "__main__":
    app = App()
    app.start()
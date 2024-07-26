import tkinter as tk
from tkinter import messagebox
import os
from PIL import Image
import numpy as np
import cv2
import mysql.connector

# Update the path to the Haar Cascade XML file
cascade_path = r"C:/Users/ADITYA JOSHI/Downloads/face identifier/haarcascade_frontalface_default.xml"

# Check if the cascade file exists
if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"{cascade_path} not found")

# Initialize Tkinter window
window = tk.Tk()
window.title("Face Recognition System")

# GUI Labels and Entry Fields
l1 = tk.Label(window, text="Name", font=("Algerian", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

def train_classifier():
    data_dir = "C:/Users/ADITYA JOSHI/Downloads/face identifier/data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')]
    faces = []
    ids = []

    if not path:
        messagebox.showerror('Error', 'No images found in the dataset directory.')
        return

    for image in path:
        try:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split(".")[1])
            faces.append(imageNp)
            ids.append(id)
            print(f"Image found: {image}, ID: {id}")
        except Exception as e:
            print(f"Error processing image {image}: {e}")

    if len(faces) == 0 or len(ids) == 0:
        messagebox.showerror('Error', 'No faces found in the dataset. Ensure the dataset is not empty and contains correctly named image files.')
        return

    ids = np.array(ids)

    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("C:/Users/ADITYA JOSHI/Downloads/face identifier/classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed!!!')

b1 = tk.Button(window, text="Training", font=("Algerian", 20), bg='orange', fg='red', command=train_classifier)
b1.grid(column=0, row=4)

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        coords = []

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            try:
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    passwd="",
                    port="4306",
                    database="Authorized_user"
                )
                mycursor = mydb.cursor()
                mycursor.execute("SELECT name FROM my_table WHERE id=%s", (id,))
                s = mycursor.fetchone()
                if s:
                    s = ''.join(s)
                else:
                    s = "Unknown"
            except mysql.connector.Error as err:
                print(f"Error: {err}")
                s = "Error"

            print(f"ID: {id}, Prediction: {pred}, Confidence: {confidence}, Name: {s}")

            if confidence > 80:  # Increase the threshold for better accuracy
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier(cascade_path)
    if faceCascade.empty():
        raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("C:/Users/ADITYA JOSHI/Downloads/face identifier/classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

b2 = tk.Button(window, text="Detect the face", font=("Algerian", 20), bg='green', fg='white', command=detect_face)
b2.grid(column=1, row=4)

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            port="4306",
            database="Authorized_user"
        )
        mycursor = mydb.cursor()

        # Create the table if it doesn't exist
        mycursor.execute("""
            CREATE TABLE IF NOT EXISTS my_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Name VARCHAR(50),
                Age INT,
                Address VARCHAR(50)
            )
        """)

        mycursor.execute("SELECT MAX(id) FROM my_table")
        max_id = mycursor.fetchone()[0]
        if max_id is None:
            max_id = 0
        new_id = max_id + 1

        sql = "INSERT INTO my_table(id, Name, Age, Address) VALUES(%s, %s, %s, %s)"
        val = (new_id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier(cascade_path)
        if face_classifier.empty():
            raise ValueError(f"Failed to load cascade classifier from {cascade_path}")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
                return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0
        save_path = "C:/Users/ADITYA JOSHI/Downloads/face identifier/data"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        while True:
            ret, frame = cap.read()
            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"{save_path}/user.{new_id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Cropped face", face)
                if cv2.waitKey(1) == 13 or img_id == 200:
                    break
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!!')

b3 = tk.Button(window, text="Generate dataset", font=("Algerian", 20), bg='pink', fg='black', command=generate_dataset)
b3.grid(column=2, row=4)

window.geometry("800x200")
window.mainloop()

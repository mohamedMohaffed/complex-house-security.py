import cv2
import os
import numpy as np
import pickle
import firebase_admin
from firebase_admin import credentials, storage
from cvzone.FaceDetectionModule import FaceDetector
import face_recognition
import cvzone
from datetime import datetime
import random
import math
from ultralytics import YOLO
class_name=['device', 'live', 'mask', 'photo']
model=YOLO("fakevsrealface.pt")


now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
def fetch_student_image(student_id):
    """
    Fetch student image from Firebase storage.
    """
    blob = bucket.get_blob(f'Images/{student_id}.jpg')
    if blob:
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        img_student = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return img_student
    else:
        print(f"Image for student ID {student_id} not found.")
        return None
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceface-f5d7f-default-rtdb.firebaseio.com/",
    'storageBucket': 'faceface-f5d7f.appspot.com'
})
bucket = storage.bucket()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize FaceDetector
face_detector = FaceDetector(minDetectionCon=0.05, modelSelection=0)

# Load known encodings and student IDs
with open('Encoding.p', "rb") as file:
    encodings_with_ids = pickle.load(file)
known_encodings, student_ids = encodings_with_ids
def main():
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break
        res = model(img)
        for r in res:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if class_name[cls] == 'live' and conf > 0.3:
                    img, bboxs = face_detector.findFaces(img)
                    if bboxs:
                        for bbox in bboxs:
                            x, y, w, h = bbox['bbox']
                            face = img[y:y + h, x:x + w]

                            # Check if the face region is valid
                            if face.size == 0:
                                print("Warning: Detected face region is empty.")
                                continue

                            try:
                                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            except cv2.error as e:
                                print(f"Error converting face to RGB: {e}")
                                continue

                            encodings_cur_frame = face_recognition.face_encodings(face_rgb)

                            if encodings_cur_frame:
                                for encoding in encodings_cur_frame:
                                    matches = face_recognition.compare_faces(known_encodings, encoding)
                                    face_distances = face_recognition.face_distance(known_encodings,
                                                                                    encoding)
                                    best_match_index = np.argmin(face_distances)

                                    if matches[best_match_index]:
                                        student_id = student_ids[best_match_index]
                                        print(f"Match found with ID: {student_id}")
                                        student_img = fetch_student_image(student_id)
                                        # if student_img is not None:
                                        # cv2.imshow('Student Image', student_img)
                                        cvzone.putTextRect(img, 'I know this face', (x, y - 10), 1, 2)
                                        print('------------------------------------------'
                                              '---------------------------------------')
                                    else:
                                        cvzone.putTextRect(img, 'I dont know this face', (x, y - 10), 1, 2)
                                        inputt = input(
                                            'do you knew this person and you want to add hem to list peaple know \n YES  NO  : ')
                                        if inputt == 'yes':
                                            random_number = random.randint(1000, 9999)
                                            cv2.imwrite(f"Images/{random_number}.jpg", face)
                                            flodersPath = "Images"
                                            PathList = os.listdir(flodersPath)
                                            imgList = []  # convort to number
                                            studentIds = []
                                            for path in PathList:
                                                imgList.append(cv2.imread(os.path.join(flodersPath, path)))
                                                studentIds.append(os.path.splitext(path)[0])
                                                fileName = f'{flodersPath}/{path}'
                                                bucket = storage.bucket()
                                                blob = bucket.blob(fileName)
                                                blob.upload_from_filename(fileName)
                                            print(studentIds)

                                            def findencodings(imgs):
                                                encodelist = []
                                                for img in imgs:
                                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                                    encode = face_recognition.face_encodings(img)[0]
                                                    encodelist.append(encode)
                                                return encodelist

                                            encodelistknew = findencodings(imgList)
                                            encodelistknewwithids = [encodelistknew, studentIds]

                                            file = open('Encoding.p', "wb")
                                            pickle.dump(encodelistknewwithids, file)
                                            file.close()
                                        elif inputt == 'no':
                                            print('you should call plaice 911')
                                        else:
                                            print('please enter yes or no')
                elif class_name[cls] == 'device' or 'mask' or 'photo' and conf > 0.3:
                    print('========================================')
                    print(f'someone try to use {class_name[cls]}')
                    print('========================================')

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


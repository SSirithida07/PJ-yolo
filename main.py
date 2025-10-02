import cv2
import os
import json
import numpy as np

PHOTO_DIR = "photo/"
HAAR_MODEL = "haarcascade_frontalface_default.xml"
FACE_SIZE = (100, 100)

# โหลด Haar Cascade
face_cascade = cv2.CascadeClassifier(HAAR_MODEL)
if face_cascade.empty():
    print(f"🚨 Error: ไม่พบไฟล์ {HAAR_MODEL}")
    exit()

# --- โหลด JSON ---
with open("students.json", "r", encoding="utf-8") as f:
    students = json.load(f)

# --- สร้างฐานข้อมูลจาก JSON ---
known_faces = {}

for student in students:
    file = student["file"]
    path = os.path.join(PHOTO_DIR, file)

    if os.path.exists(path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = gray[y:y+h, x:x+w]
            face_standard = cv2.resize(roi, FACE_SIZE)
            # เก็บเป็น key = student["id"]
            known_faces[student["id"]] = {
                "name": student["name"],
                "face": face_standard
            }
        else:
            print(f"⚠️ ไม่พบใบหน้าใน {file}")
    else:
        print(f"❌ ไม่พบไฟล์รูป {file}")

# --- Real-Time Recognition ---
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_in_frame = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(30,30))

    for (x, y, w, h) in faces_in_frame:
        roi = gray_frame[y:y+h, x:x+w]
        face_standard = cv2.resize(roi, FACE_SIZE)

        best_match_id = None
        best_score = float('inf')

        # เปรียบเทียบกับฐานข้อมูล JSON
        for sid, data in known_faces.items():
            diff = cv2.absdiff(data["face"], face_standard)
            score = np.mean(diff)
            if score < best_score:
                best_score = score
                best_match_id = sid

        # แสดงผล
        if best_score < 60:
            name = known_faces[best_match_id]["name"]
            color = (0,255,0)
        else:
            name = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({best_score:.1f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Match (JSON)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

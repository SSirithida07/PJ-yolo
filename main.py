import cv2
import os
import numpy as np

# 📂 โฟลเดอร์เก็บรูป
PHOTO_DIR = "photo/"
HAAR_MODEL = "haarcascade_frontalface_default.xml"
FACE_SIZE = (100, 100) # กำหนดขนาดมาตรฐานสำหรับเปรียบเทียบภาพ

# --- โหลด Haar Cascade Model ---
face_cascade = cv2.CascadeClassifier(HAAR_MODEL)
if face_cascade.empty():
    print(f"🚨 Error: ไม่พบไฟล์ {HAAR_MODEL} กรุณาวางไว้ในโฟลเดอร์เดียวกัน")
    exit()

# --- สร้างฐานข้อมูลสำหรับเปรียบเทียบภาพ ---
known_faces = {}

for file in os.listdir(PHOTO_DIR):
    path = os.path.join(PHOTO_DIR, file)
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. ตรวจจับใบหน้า
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # 2. ตัดภาพเฉพาะใบหน้าและปรับขนาดมาตรฐาน
            face_roi = gray_image[y:y+h, x:x+w]
            face_standard = cv2.resize(face_roi, FACE_SIZE)
            
            # 3. เก็บใบหน้าที่ปรับแล้วในฐานข้อมูล (ใช้ไฟล์ชื่อเป็นคีย์)
            known_faces[file] = face_standard
        else:
            print(f"⚠️ ไม่พบใบหน้าใน {file}")


# --- Real-Time Recognition ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับใบหน้าในเฟรมปัจจุบัน
    faces_in_frame = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces_in_frame:
        # ตัดภาพใบหน้าและปรับขนาด
        current_face_roi = gray_frame[y:y+h, x:x+w]
        current_face_standard = cv2.resize(current_face_roi, FACE_SIZE)

        best_match_name = "Unknown"
        best_score = float('inf') # ค่าความแตกต่าง (ต่ำสุดคือดีที่สุด)
        
        # เปรียบเทียบใบหน้าปัจจุบันกับทุกใบหน้าในฐานข้อมูล
        for name, known_face in known_faces.items():
            # ใช้เทคนิคเปรียบเทียบ histogram หรือ L2 norm (ความแตกต่างของพิกเซล)
            # ในตัวอย่างนี้ใช้ค่าความแตกต่างเฉลี่ยของพิกเซล (ง่ายที่สุด)
            diff = cv2.absdiff(known_face, current_face_standard)
            score = np.mean(diff)
            
            if score < best_score:
                best_score = score
                best_match_name = name

        # กำหนดเกณฑ์: หากค่าความแตกต่างต่ำกว่า 60 (ตัวอย่าง) ถือว่า "จำได้"
        if best_score < 60: 
            display_name = best_match_name
            color = (0, 255, 0) # สีเขียว
        else:
            display_name = "Unknown"
            color = (0, 0, 255) # สีแดง

        # แสดงผล
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{display_name} (Score: {best_score:.1f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    cv2.imshow("OpenCV Simple Face Match", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
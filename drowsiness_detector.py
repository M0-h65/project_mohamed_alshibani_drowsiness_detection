import cv2
import mediapipe as mp
import numpy as np
import pygame
import datetime
import os

# تعريف مؤشرات العين والفم
# هذه المؤشرات تعتمد على ترتيب النقاط التي يوفرها MediaPipe Face Mesh
# يرجى الرجوع إلى وثائق MediaPipe Face Mesh للحصول على الترتيب الدقيق
# كمثال، هذه أرقام تقريبية للعين اليسرى واليمنى والفم
# ستحتاج إلى التحقق من هذه الأرقام وتعديلها بناءً على مخرجات MediaPipe
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # نقاط العين اليسرى الصحيحة لحساب EAR
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # نقاط العين اليمنى الصحيحة لحساب EAR
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61] # مثال

# دالة لحساب Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # حساب المسافات الإقليدية بين النقاط العمودية والأفقية للعين
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# دالة لحساب Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    # حساب المسافات الإقليدية بين النقاط العمودية والأفقية للفم
    A = np.linalg.norm(np.array(mouth[13]) - np.array(mouth[19]))
    B = np.linalg.norm(np.array(mouth[14]) - np.array(mouth[18]))
    C = np.linalg.norm(np.array(mouth[15]) - np.array(mouth[17]))
    D = np.linalg.norm(np.array(mouth[12]) - np.array(mouth[16]))
    mar = (A + B + C) / (2.0 * D)
    return mar

# عتبات الكشف عن النعاس والتثاؤب
EYE_AR_THRESH = 0.25  # عتبة أعلى لاكتشاف النعاس بسهولة
EYE_AR_CONSEC_FRAMES = 5  # عدد إطارات أقل لتنبيه أسرع
MOUTH_AR_THRESH = 0.7

# عدادات الإطارات
COUNTER = 0
YAWN_COUNTER = 0

# تهيئة MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# بدء التقاط الفيديو من الكاميرا
cap = cv2.VideoCapture(0)

# تهيئة pygame للصوت
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# متغيرات حالة تسجيل الأحداث
last_drowsy_logged = False
last_yawn_logged = False

# إنشاء مجلدات للقطات النعاس والتثاؤب إذا لم تكن موجودة
if not os.path.exists("snapshots/drowsy"):
    os.makedirs("snapshots/drowsy")
if not os.path.exists("snapshots/yawn"):
    os.makedirs("snapshots/yawn")


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # قلب الصورة أفقيًا لعرض سيلفي، وتحويلها إلى RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # رسم نقاط الوجه
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec, drawing_spec)

            # استخراج إحداثيات نقاط العين والفم
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmarks.append((x, y))

            left_eye = [landmarks[i] for i in LEFT_EYE_LANDMARKS]
            right_eye = [landmarks[i] for i in RIGHT_EYE_LANDMARKS]
            print(f"left_eye: {left_eye}")
            print(f"right_eye: {right_eye}")
            mouth = [landmarks[i] for i in MOUTH_LANDMARKS]

            # حساب EAR و MAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            print(f"EAR: {ear:.3f}")  # طباعة قيمة EAR في كل إطار
            mar = mouth_aspect_ratio(mouth)

            # الكشف عن النعاس
            # لا داعي لتعريف global هنا لأن المتغيرات معرفة خارج الحلقة
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(image, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # تكرار الصوت باستمرار
                    if not pygame.mixer.get_busy():
                        alert_sound.play()
                    # تسجيل وقت النعاس في ملف وأخذ لقطة
                    if not last_drowsy_logged:
                        now = datetime.datetime.now()
                        with open("event_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"نعاس - {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        # حفظ لقطة
                        snapshot_name = f"snapshots/drowsy/drowsy_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(snapshot_name, image)
                        last_drowsy_logged = True
            else:
                COUNTER = 0
                last_drowsy_logged = False

            # الكشف عن التثاؤب
            if mar > MOUTH_AR_THRESH:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= 5:
                    cv2.putText(image, "YAWNING!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    # تسجيل التثاؤب في ملف وأخذ لقطة
                    if not last_yawn_logged:
                        now = datetime.datetime.now()
                        with open("event_log.txt", "a", encoding="utf-8") as f:
                            f.write(f"تثاؤب - {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        snapshot_name = f"snapshots/yawn/yawn_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(snapshot_name, image)
                        last_yawn_logged = True
            else:
                YAWN_COUNTER = 0
                last_yawn_logged = False

            # عرض EAR و MAR على الشاشة
            cv2.putText(image, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"MAR: {mar:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # تكبير الصورة قبل العرض
    display_image = cv2.resize(image, (640, 480))
    cv2.imshow('Driver Drowsiness Detection', display_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



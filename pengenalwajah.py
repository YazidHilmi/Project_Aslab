# Import semua library yang dibutuhkan
import cv2
import numpy as np

face_ref = cv2.CascadeClassifier("face_ref.xml")
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

faceDir = 'datawajah'
latihDir = 'LatihWajah' 

faceRecognizer.read(latihDir + '/training.xml')
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

names = {51423574: "Yazid Hilmi", 12345678: "Chico"}

minWidth = 0.1 * camera.get(3)
minHeight = 0.1 * camera.get(4)

def main():
    while True:
        _, frame = camera.read()
        # Ubah frame ke grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        # Deteksi wajah pada frame grayscale
        kotak_scan(frame, gray_frame)
        
        cv2.imshow('Deteksi Wajah', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            keluar()
            break

def kotak_scan(frame, gray_frame):
    faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=2, minSize=(round(minWidth), round(minHeight)))
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
        
        # Ambil bagian wajah dari frame grayscale
        face_img = gray_frame[y:y+h, x:x+w]
        id, confidence = faceRecognizer.predict(face_img)
        print(id, confidence)
        
        # Jika confidence kecil, maka data cocok
        if confidence <= 50:  
            nameID = names.get(id, "Tidak Diketahui")
            confidenceTxt = "{0}%".format(round(100 - confidence))
        else:
            nameID = "Tidak Diketahui"
            confidenceTxt = "{0}%".format(round(100 - confidence))

        
        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidenceTxt), (x+5, y+h-5), font, 1, (255, 255, 0), 2)

def keluar():
    camera.release()
    cv2.destroyAllWindows()
    exit()

if __name__ == "__main__":
    main()

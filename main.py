import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
smile = cv2.CascadeClassifier("smile.xml")
camera = cv2.VideoCapture(0)

faceID = input("Masukkan NIM : ")
print("Lihat Kamera...")
wajahDir = 'datawajah'


def smile_detection(frame):
    optimum_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    senyum = smile.detectMultiScale(optimum_frame, scaleFactor = 1.1, minNeighbors = 3,minSize=(300,300))
    return senyum

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor = 1.2, minNeighbors = 3, minSize=(800,800))
    return faces

def drawer_box(frame, faceID, ambilData):
    for x, y, w, h in smile_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        namaFile = 'wajah.'+ str(faceID) + '.' + str(ambilData) + '.jpg'
        cv2.imwrite(wajahDir + '/' + namaFile, frame)
        ambilData += 1
        if ambilData >= 30:
            return ambilData
        else:
            continue
    return ambilData
        

def close_window():
    camera.release
    cv2.destroyAllWindows()
    print ("Pengambilan Data Wajah Selesai")
    exit()

def main():
    ambilData = 1
    while True:
        _, frame = camera.read()
        ambilData = drawer_box(frame, faceID, ambilData)
        cv2.imshow("Wajah Deteksi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or ambilData >=30 :
            return close_window
        
    
if __name__ == '__main__':
    main()
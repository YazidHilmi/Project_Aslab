import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
smile = cv2.CascadeClassifier("smile.xml")
camera = cv2.VideoCapture(0)
 
def smile_detection(frame):
    optimum_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    senyum = smile.detectMultiScale(optimum_frame, scaleFactor = 1.1, minNeighbors = 3,minSize=(500,500))
    return senyum

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor = 1.4, minNeighbors = 3, minSize=(1000,1000))
    return faces

def drawer_box(frame):
    for x, y, w, h in smile_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

def close_window():
    camera.release
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("Wajah Deteksi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
    
if __name__ == '__main__':
    main()
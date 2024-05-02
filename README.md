# face-recognation

import cv2

##kamerayı açmak için bu fonksiyonu kullanıyoryuz
cm = cv2.VideoCapture(0)

##kamerada görülecek kırmızı alanı tanımlıyoruz ardından alanın sınırlarını belirliyıoruz
show_red_area = True
red_area = (0, 0, 640, 640)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

startTime = None
endTime = None

while True:
    ret, frame = cm.read()
    if not ret:
        break

    if show_red_area:

        x, y, c, v = red_area
        cv2.rectangle(frame, (x, y), (x + c, y + v), (0, 0, 255), 2)


    gray = cv2.cvtColor(frame[y:y + v, x:x + c], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    ## global fonskiyonu ile elapsed_time ı heryerden erişilebilir yapıyoruz aksi takdırde puttext hata verebiliyor
    global elapsed_time
    if len(faces) > 0:
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x + x1, y + y1), (x + x1 + x2, y + y1 + y2), (0, 255, 0), 2)

        if startTime is None:
            startTime = cv2.getTickCount()
        endTime = None
        show_red_area = False
    else:
        if endTime is None and startTime is not None:
            endTime = cv2.getTickCount()
            elapsed_time = (endTime - startTime) / cv2.getTickFrequency()
            print("Yüz algılanma süresi: {:.2f} saniye".format(elapsed_time))

            startTime = None
            endTime = None
            show_red_area = True


        ##kamerada yüzün kaldığı zamanı yazdırmak için puttext fonksiyonu kullanıyoruz
        cv2.putText(frame, f'Time: {elapsed_time:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cm.release()
cv2.destroyAllWindows()

import cv2                        # Görüntü alma, gösterme, çizim işlemlerinin olduğu kütüp yani opencv nin phyton versiyonu
import dlib                      # Yüz tespiti ve landmark (68 nokta) tespiti fonksiyonlarının olduğu kütüp
from scipy.spatial import distance  # Öklidyen mesafe hesaplamak için gerekli 
from imutils import face_utils   # Dlib'deki şekli numpy array'e çevirmek için gerekli 

# Göz açıklık oranını (EAR) hesaplayan fonksiyon
def eye_aspect_ratio(eye):
    # Gözün üst ve alt kapağındaki mesafeleri 
    A = distance.euclidean(eye[1], eye[5])  # Üst-alt 1. dikey mesafe
    B = distance.euclidean(eye[2], eye[4])  # Üst-alt 2. dikey mesafe
    # Gözün yatay uzunluğunu
    C = distance.euclidean(eye[0], eye[3])  # Gözün yatay uzunluğu
    # EAR (Eye Aspect Ratio) formülü
    ear = (A + B) / (2.0 * C)
    return ear  

# Dlib'in yüz dedektörünü ve 68 noktalı landmark modeli
detector = dlib.get_frontal_face_detector()  # Yüz algılayıcı (kutu içinde yüz tespiti)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Noktaları tespit edecek model

# Eşik değerler ve sayaçlar
EYE_AR_THRESH = 0.25          # Threshold değerini 0.25 belirledim yani göz açıklığı bu değerin altına düşerse, göz kapalı kabul edilir
EYE_AR_CONSEC_FRAMES = 3      # Göz arka arkaya 3 frame boyunca kapalıysa blink sayılır
COUNTER = 0                   # Arka arkaya kaç frame boyunca göz kapalı sayıldı
TOTAL = 0                     # Toplam göz kırpma sayısı

# Kamera başlat
cap = cv2.VideoCapture(0)

# yüz algılayıp göz kırpmasını kontrol etmece
while True:
    ret, frame = cap.read()  # Kameradan görüntü almaca
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Renkli görüntüyü griye çevirdim çünkü yüz tespiti için daha verimli oluyor aksi halde RGB de iş karışabiliyor.
    faces = detector(gray, 0)  # Yüzleri tespit etmece

    for face in faces:
        shape = predictor(gray, face)    # Yüzdeki 68 noktayı tespit etmece
        shape = face_utils.shape_to_np(shape)       
        # yüzün üzerindeki noktalar daha kolay kullanılsın diye diziye(numpy array) çevirmek gerek , daha sonraki işlemlerdeki yapı düzenli yapı istiyor aksi halde kod çalışmıyor denedim olmadı.

        leftEye = shape[42:48]                      # Sol gözün landmark noktaları
        rightEye = shape[36:42]                     # Sağ gözün landmark noktaları

        leftEAR = eye_aspect_ratio(leftEye)         
        rightEAR = eye_aspect_ratio(rightEye)       
        # Her iki göz için de ayrı ayrı EAR hesaplamaca
        
        ear = (leftEAR + rightEAR) / 2.0       
        # Ortalamasını alıyoruz burada ama gerçek koşullarda kişinin göz yapısına göre sağ veya sol gözün ağırlık katsayıları değiştirilebilir mesela benim sol göz kapağım daha düşük bu yüzden katsayılı formül daha etkili olabilir 

        if ear < EYE_AR_THRESH:                     # ear thrshold değerinden küçükse göz kapalı
            COUNTER += 1                            # Sayaç artır
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:     # Eğer göz uzun süre kapalı kaldıysa
                TOTAL += 1                          # Bir göz kırpması olarak say , 
                #Yani adam bir kez gözünü kırptı diye hemen bunun uykusu var demiyoruz insan göz kırpabilir ancak bu uzun süreli olduğunda yorgunluk algılatıyoruz
            COUNTER = 0                             # Sayaç sıfırlanır , sıfırlamazsak uyarı verdikten sonra sürekli olarak uyarı vermeye devam eder durduramayız onu
            

        # Toplam blink sayısını ekrana yazmaca
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Eye Blink Detection", frame)        # Görüntüyü pencereye yazmaca

    if cv2.waitKey(1) & 0xFF == ord("q"):           # Q çıkış tuşu olarak ayarlamaca
        break

# Temizlik işlemleri
cap.release()                    # Kamerayı kapat
cv2.destroyAllWindows()          

import cv2
import numpy as np

# Definir los rangos de color amarillo en el espacio de color HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Capturar video en tiempo real desde la cámara
cap = cv2.VideoCapture(0)

print("Iniciando detector de color amarillo...")
while True:
    # Leer un frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear una máscara para el color amarillo
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Mostrar los resultados
    cv2.imshow("Video Original", frame)
    cv2.imshow("Detector de Color Amarillo", result)
    
    # Salir del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Detector de color amarillo finalizado.")

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

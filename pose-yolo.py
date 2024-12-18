from ultralytics import YOLO
import cv2

# Carga del modelo YOLO
model = YOLO("yolo11n-pose.pt")

# Etiquetas de los puntos clave
keypoint_labels = [
    "Nariz", "Ojo izquierdo", "Ojo derecho", "Oreja izquierda", "Oreja derecha",
    "Hombro izquierdo", "Hombro derecho", "Codo izquierdo", "Codo derecho",
    "Muñeca izquierda", "Muñeca derecha", "Cadera izquierda", "Cadera derecha",
    "Rodilla izquierda", "Rodilla derecha", "Tobillo izquierdo", "Tobillo derecho"
]

# Captura desde la cámara
cap = cv2.VideoCapture(0)

# Procesamiento en tiempo real
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara.")
        break

    # Predicción y obtención de resultados
    results = model.predict(frame, save=False, show=False)
    annotated_frame = frame.copy()

    # Bandera para verificar si ambos ojos están detectados
    both_eyes_detected = False

    # Procesar puntos clave por persona
    for person_keypoints in results[0].keypoints.data:  # Acceder al tensor
        eye_left_conf = person_keypoints[1][2].item()  # Confianza del ojo izquierdo
        eye_right_conf = person_keypoints[2][2].item()  # Confianza del ojo derecho

        if eye_left_conf > 0.8 and eye_right_conf > 0.8:  # Verificar confianza
            both_eyes_detected = True

            # Dibujar puntos clave si la confianza es suficiente
            for i, keypoint in enumerate(person_keypoints):
                x, y, conf = keypoint.tolist()  # Extraer valores
                if conf > 0.8:  # Dibujar solo puntos clave confiables
                    label = keypoint_labels[i]
                    # Dibujar el nombre del punto clave en el marco
                    cv2.putText(
                        annotated_frame, label, (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Detener si no se detectaron ambos ojos
    if not both_eyes_detected:
        print("Ambos ojos no detectados con suficiente confianza. Finalizando...")
        break

    # Mostrar el frame en tiempo real
    cv2.imshow("Real-Time Pose Detection", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

import cv2
import time
import mediapipe as mp

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3)

estado_bess = False
bess_ativado_em = None
bess_descarregado = False
protocolo_ativo = False
detectou_rosto_na_emergencia = None
luzes_ativas = False

cap = cv2.VideoCapture(0)
ultimo_print_brilho = 0

tempo_ultima_verificacao = None  # nova variável para controle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brilho_medio = gray.mean()
    tempo_atual = time.time()

    if tempo_atual - ultimo_print_brilho >= 5:
        print(f"[DEBUG] Brilho médio: {brilho_medio:.2f}")
        ultimo_print_brilho = tempo_atual

    resultado_face = detector.process(frame_rgb)
    rosto_detectado = resultado_face.detections is not None and len(resultado_face.detections) > 0

    if rosto_detectado:
        for det in resultado_face.detections:
            bboxC = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = x1 + int(bboxC.width * w)
            y2 = y1 + int(bboxC.height * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if brilho_medio < 80:
        if not estado_bess:
            estado_bess = True
            bess_ativado_em = tempo_atual
            bess_descarregado = False
            protocolo_ativo = False
            detectou_rosto_na_emergencia = None
            luzes_ativas = False
            print("[INFO] BESS ativado!")
        else:
            if (tempo_atual - bess_ativado_em >= 10) and not bess_descarregado:
                bess_descarregado = True
                protocolo_ativo = True
                print("[ALERTA] Bess descarregado - ativando protocolo de emergência")
    else:
        if estado_bess or protocolo_ativo or luzes_ativas:
            print("[INFO] BESS carregando — Luzes de emergência desligadas")
        estado_bess = False
        bess_descarregado = False
        protocolo_ativo = False
        detectou_rosto_na_emergencia = None
        luzes_ativas = False

    # Verificação do rosto durante protocolo ativo, a cada 5 segundos (ou primeira vez)
    if protocolo_ativo:
        if detectou_rosto_na_emergencia is None or (tempo_ultima_verificacao and tempo_atual - tempo_ultima_verificacao > 5):
            if rosto_detectado:
                detectou_rosto_na_emergencia = True
                luzes_ativas = True
                print("[ALERTA] Rosto detectado — Luzes de emergência ativadas!")
            else:
                detectou_rosto_na_emergencia = False
                print("[ALERTA] Nenhum rosto detectado — aguardando presença")
            tempo_ultima_verificacao = tempo_atual

    if luzes_ativas:
        print("[INFO] Luzes de emergência estão ligadas")
        luzes_ativas = False

    cv2.imshow("Monitoramento com Detecção Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

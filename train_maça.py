from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
from pathlib import Path

# Função para leitura de imagens que lida com caminhos Unicode
def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    """
    Read an image from a file.

    Args:
        filename (str): Path to the file to read.
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

    Returns:
        (np.ndarray): The read image.
    """
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)

# Define o caminho da imagem
image_path = r"C:\Users\jc018\Downloads\datasets\image.png"
# Carrega a imagem usando a função imread
img = imread(image_path)

# Verifica se a imagem foi carregada corretamente
if img is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    exit()

# Usa modelo da Yolo
# Model	    size    mAPval  Speed       Speed       params  FLOPs
#           (pixels) 50-95  CPU ONNX A100 TensorRT   (M)     (B)
#                           (ms)        (ms)
# YOLOv8n	640	    37.3	80.4	    0.99	    3.2	    8.7
# YOLOv8s	640	    44.9	128.4	    1.20	    11.2	28.6
# YOLOv8m	640	    50.2	234.7	    1.83	    25.9	78.9
# YOLOv8l	640	    52.9	375.2	    2.39	    43.7	165.2
# YOLOv8x	640	    53.9	479.1	    3.53	    68.2	257.8

#model = YOLO("yolov8n.pt")

# Meu  modelo
model = YOLO("runs/detect/train3/weights/best.pt")

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

while True:
    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Processa a lista de resultados
    for result in results:
        # Visualiza os resultados na imagem
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                # Obtém as caixas e os IDs de rastreamento
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Desenha as trilhas
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # ponto central x, y
                    if len(track) > 30:  # mantém 30 trilhas para 30 frames
                        track.pop(0)

                    # Desenha as linhas de rastreamento
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except:
                pass

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
print("Desligando")

import sys
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QScrollArea
from tensorflow import keras

class FrameProcessor(QObject):
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, camera_index, model_path):
        super().__init__()
        self.camera_index = camera_index
        self.camera = cv2.VideoCapture(camera_index)
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        self.is_running = True

    def process_frame(self):
        while self.is_running:
            _, frame = self.camera.read()

            im = Image.fromarray(frame, 'RGB')
            im = im.resize((224, 224))
            img_array = image.img_to_array(im)
            img_array = np.expand_dims(img_array, axis=0) / 255
            probabilities = self.model.predict(img_array)[0]
            prediction = np.argmax(probabilities)

            if (prediction == 1) and (probabilities[prediction] >= 0.7):
                lower_red = np.array([0, 0, 100])
                upper_red = np.array([100, 100, 255])
                mask = cv2.inRange(frame, lower_red, upper_red)
                frame = cv2.bitwise_and(frame, frame, mask=mask)
                print(probabilities[prediction])
            else:
                print(f"FIRE: {probabilities[prediction]}")

            self.frame_processed.emit(frame)

        self.release_camera()

    def release_camera(self):
        self.camera.release()

class CameraWindow(QMainWindow):
    def __init__(self, camera_index, model_path):
        super().__init__()
        self.processor = FrameProcessor(camera_index, model_path)
        self.setWindowTitle(f"Camera {camera_index}")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # QLabel для отображения изображения
        self.image_label = QLabel()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)  # Разрешаем масштабирование содержимого
        self.layout.addWidget(self.scroll_area)

        # Кнопка для увеличения/сворачивания окна
        self.toggle_button = QPushButton("Toggle Window")
        self.toggle_button.clicked.connect(self.toggle_window)
        self.layout.addWidget(self.toggle_button)

        self.is_maximized = False  # Флаг для отслеживания состояния окна

        self.thread = QThread()
        self.processor.moveToThread(self.thread)
        self.thread.started.connect(self.processor.process_frame)
        self.processor.frame_processed.connect(self.update_frame)
        self.thread.start()

    def toggle_window(self):
        if self.is_maximized:
            self.showNormal()  # Восстановить размер окна
        else:
            self.showMaximized()  # Увеличить окно до максимального размера
        self.is_maximized = not self.is_maximized  # Инвертировать состояние флага

    def update_frame(self, frame):
        q_image = self.convert_frame_to_qimage(frame)
        # Масштабируем изображение до размера окна
        q_image = q_image.scaled(self.scroll_area.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        # Устанавливаем фиксированный размер для QLabel
        self.image_label.setFixedSize(q_image.size())

    def convert_frame_to_qimage(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        return QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def closeEvent(self, event):
        self.processor.is_running = False
        self.processor.release_camera()
        self.thread.quit()
        self.thread.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    windows = []
    for camera_index in range(2):
        window = CameraWindow(camera_index, 'model_2.h5')
        window.show()
        windows.append(window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

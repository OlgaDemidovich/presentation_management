import sys
import numpy as np
import cv2
import time

import mediapipe as mp

import pyautogui as pag
import threading

from PyQt5.QtWidgets import QApplication, QMainWindow
from settings_ui import Ui_Settings
from reference_ui import Ui_Reference


class SettingsWidget(QMainWindow, Ui_Settings):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.video_play = False
        self.point_need = False
        self.btn_push = False
        self.running = True
        self.pointers, self.managers = 'Right', 'Left'
        self.need_pointer.stateChanged.connect(self.pointer)
        self.apply_btn.clicked.connect(self.apply)
        self.reference.clicked.connect(self.reference_click)

    def reference_click(self):
        self.ex2 = ReferenceWidget(self)
        self.ex2.show()

    def pointer(self):
        if self.need_pointer.isChecked():
            self.pointer_and_manager.setEnabled(True)
            self.commands.setEnabled(True)
            self.manager.setEnabled(False)
        else:
            self.pointer_and_manager.setEnabled(False)
            self.commands.setEnabled(False)
            self.manager.setEnabled(True)

    def apply(self):
        self.btn_push = True
        if self.need_pointer.isChecked():
            if self.pointer_and_manager.currentText() == 'Правая рука для указки, левая для управления':
                self.pointers = 'Right'
                self.managers = 'Left'
            else:
                self.pointers = 'Left'
                self.managers = 'Right'
        else:
            if self.manager.currentText() == 'Правая рука для управления':
                self.pointers = 'Left'
                self.managers = 'Right'
            else:
                self.pointers = 'Right'
                self.managers = 'Left'

        self.video_play = self.video.isChecked()
        self.point_need = self.need_pointer.isChecked()
        self.need_command = self.commands.currentText()
        _settings()
        # self.close()

    def setting(self):
        return self.video_play, self.point_need, self.pointers, self.managers, self.need_command


class ReferenceWidget(QMainWindow, Ui_Reference):
    def __init__(self, parent=None):
        super(ReferenceWidget, self).__init__(parent)
        self.setupUi(self)


def __closed():
    global running
    running = False


def _settings():
    global videoEnabled, need_pointer, hand_name_pointer, hand_name_manager, command
    videoEnabled, need_pointer, hand_name_pointer, hand_name_manager, command = ex.setting()


def __board_calculation():
    global camera_picture, ih, iw
    image = camera_picture
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ih, iw, ic = image.shape
    if results.multi_face_landmarks:

        all_xy = np.array(
            [[i.x, i.y] for i in results.multi_face_landmarks[0].landmark])
        all_x = all_xy[:, 0] * iw
        all_y = all_xy[:, 1] * ih
        x_nose, y_nose = int(all_x[4]), int(all_y[4])
        x_min, x_max, y_min, y_max = int(np.min(all_x)), int(
            np.max(all_x)), int(np.min(all_y)), int(np.max(all_x))

        f_w = x_max - x_min
        global h_board, w_board, x_board, y_board
        x1_board = int(x_min - f_w * 2.5)
        x2_board = int(x_max + f_w * 2.5)
        w_board = x2_board - x1_board
        h_board = w_board * height // width
        y1_board = y_nose - h_board // 2
        y2_board = y_nose + h_board // 2

        y_board = y1_board
        x_board = iw - x2_board
        if videoEnabled:
            cv2.rectangle(image, (x1_board, y1_board), (x2_board, y2_board),
                          (255, 255, 255), 1)
            cv2.rectangle(image, (
                int(x1_board + w_board * 0.3), int(y_board + h_board * 0.7)),
                          (int(x1_board + w_board * 0.7),
                           int(y_board + h_board * 0.3)), (255, 255, 255), 1)
            cv2.circle(image, (x2_board, y1_board), 5, (255, 255, 255), -1)

    camera_picture = image


def __hand_detection():
    global camera_picture, hand_name
    image = cv2.flip(camera_picture, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            landmark = list(enumerate(hand_landmarks.landmark))[8][1]
            hand_name = handedness.classification[0].label[0:]
            x, y = int(landmark.x * iw), int(landmark.y * ih)
            global x_finger, y_finger
            x_finger, y_finger = x, y
            if videoEnabled:
                if hand_name == "Right":
                    cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
                if hand_name == "Left":
                    cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

    else:
        hand_name = False

    camera_picture = image


def __cords_finger():
    global x_coordinate, y_coordinate
    x_b, y_b = x_board, y_board
    x_coordinate, y_coordinate = int((x_finger - x_b) / w_board * width), \
                                 int((y_finger - y_b) / h_board * height)


def __painter_point():
    cv2.putText(camera_picture, 'Указка влючена', (180, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(camera_picture, 'Указка влючена', (180, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


def __painter_commands():
    cv2.putText(camera_picture, signal_swipe, (20, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(camera_picture, signal_swipe, (20, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


def __get_cap(index):
    next_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    next_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    next_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    time.sleep(1)
    return next_cap


def __find_camera():
    # Выбор камеры, переданной программе в качестве аргумента
    if len(sys.argv) > 1:
        next_cap = __get_cap(int(sys.argv[1]))
        loaded, initial_frame = next_cap.read()
        if loaded and np.sum(initial_frame) > 0:
            print("Камера", sys.argv[1], "выбрана принудительно")
            return next_cap
        else:
            print("Не удалось выбрать камеру", sys.argv[1])
        next_cap.release()

    # Поиск доступной камеры с предпочтением ко внешним камерам
    for camera_index in range(5, -1, -1):
        next_cap = __get_cap(camera_index)
        loaded, initial_frame = next_cap.read()
        if loaded and np.sum(initial_frame) > 0:
            print('Камера найдена')
            return next_cap

        next_cap.release()
    return False


def __watcher():
    global camera_picture, t0, flag_pointer, t1, signal_swipe, lbm_click
    while running:
        ret, camera_picture = cap.read()
        if not ret:
            break
        __board_calculation()
        __hand_detection()
        cv2.waitKey(10)
        if need_pointer and hand_name == hand_name_pointer and flag_pointer and \
                x_board <= x_finger <= x_board + w_board and y_board <= y_finger <= y_board + h_board:
            # (iw > x_board + w_board > x_board > 0 and ih > y_board + h_board > y_board > 0) and \
            if videoEnabled:
                cv2.circle(camera_picture, (x_finger, y_finger), 5, (0, 0, 255),
                           -1)
            __cords_finger()
            pag.moveTo(x_coordinate, y_coordinate)

        if hand_name == hand_name_manager:
            if not t0 and x_board + w_board * 0.3 <= x_finger <= x_board + w_board * 0.7 and \
                    y_board + h_board * 0.3 <= y_finger <= y_board + h_board * 0.7:
                t0 = time.monotonic()

            if t0 and time.monotonic() - t0 >= 0.2:
                if x_finger < x_board + w_board * 0.3:
                    pag.hotkey('left')
                    if videoEnabled:
                        t1 = time.monotonic()
                        signal_swipe = 'Назад'
                    t0 = False
                elif x_finger > x_board + w_board * 0.7:
                    pag.hotkey('right')
                    if videoEnabled:
                        t1 = time.monotonic()
                        signal_swipe = 'Вперед'
                    t0 = False
                elif need_pointer and y_finger < y_board + h_board * 0.3 and \
                        x_board + w_board * 0.3 < x_finger < x_board + w_board * 0.7:
                    flag_pointer = not flag_pointer
                    if command == 'ctrl+l':
                        pag.hotkey("ctrl", 'l')
                    elif command == 'ctrl+ПКМ':
                        lbm_click = not lbm_click
                        if lbm_click:
                            with pag.hold('ctrl'):
                                pag.mouseDown(button='left')
                        else:
                            with pag.hold('ctrl'):
                                pag.mouseUp(button='left')
                    t0 = False
        if videoEnabled:
            if t1 and time.monotonic() - t1 <= 0.5:
                __painter_commands()
            else:
                t1 = False
            if flag_pointer:
                __painter_point()
            cv2.imshow('Frame', camera_picture)
            global video_played
            video_played = True
        elif video_played:
            video_played = False
            cv2.destroyWindow("Frame")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Настраиваемые параметры #
    videoEnabled = False  # вывод видео
    need_pointer = False  # использование указки
    hand_name_pointer, hand_name_manager = 'Right', 'Left'
    command = 'ctrl+l'
    # Конец настраиваемых параметров #

    camera_picture = False
    video_played = False
    hand_name = False
    signal_swipe = False
    need_ctrl_l, need_ctrl_lbm = False, False
    h_board, w_board, x_board, y_board = 0, 0, 0, 0
    x_finger, y_finger = 0, 0
    x_coordinate, y_coordinate = 0, 0
    words = ['вперёд', 'назад', 'указка']
    flag_pointer = False
    lbm_click = False
    t0 = False
    t1 = False
    ih, iw = False, False
    pag.FAILSAFE = False
    running = True

    # инициализация моделей для распознования ключевых точек на руках и лице ##############
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8)

    cap = __find_camera()
    if not cap:
        print("Камера не найдена")
    else:
        width, height = pag.size()
        app = QApplication(sys.argv)
        ex = SettingsWidget()
        ex.show()
        app.lastWindowClosed.connect(__closed)
        threading.Thread(target=__watcher, daemon=True).start()
        sys.exit(app.exec_())

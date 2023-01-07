import sys
import numpy as np
import cv2
import time
import copy
import csv
import itertools
import win32gui, win32ui, win32con, win32api
from time import sleep
from ctypes import windll
from PIL import Image

import mediapipe as mp

import pyautogui as pag
import threading

from PyQt5.QtWidgets import QApplication, QMainWindow
from settings_ui import Ui_Settings
from reference_ui import Ui_Reference
from model_2 import KeyPointClassifier


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


def __window_pdf_enum_callback(_hwnd, _result):
    if str(win32gui.GetClassName(_hwnd)) in ['screenClass',
                                             'protectedViewScreenClass']:
        _result.append(_hwnd)


def __window_child_callback(_hwnd, _result):
    _result.append(_hwnd)


def find_all_pdf_windows():
    _result = []
    win32gui.EnumWindows(__window_pdf_enum_callback, _result)

    return _result


def get_all_child_hwnd(_hwnd):
    _result = []
    win32gui.EnumChildWindows(_hwnd, __window_child_callback, _result)

    return _result


def get_pdf_handler(_hwnd):
    return [i for i in get_all_child_hwnd(_hwnd) + [_hwnd] if
            'PowerPoint' in win32gui.GetWindowText(i)][0]


def get_window_bitmap(_hwnd):
    left, top, right, bot = win32gui.GetWindowRect(_hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(_hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    windll.user32.PrintWindow(_hwnd, saveDC.GetSafeHdc(), 2)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    _image = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(_hwnd, hwndDC)

    return _image


def perform_click(_hwnd, toPrev):
    wParam = win32api.MAKELONG(1, win32con.WHEEL_DELTA * (1 if toPrev else -1))
    lParam = win32api.MAKELONG(10, 10)
    win32api.PostMessage(_hwnd, win32con.WM_MOUSEWHEEL, wParam, lParam)


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
    keypoint_classifier = KeyPointClassifier()
    with open('model_2/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

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

            # Bounding box calculation
            brect = calc_bounding_rect(image, hand_landmarks)
            # Landmark calculation
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            print(keypoint_classifier_labels[hand_sign_id])

    else:
        hand_name = False

    camera_picture = image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


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

        windows = find_all_pdf_windows()
        if len(windows) == 0:
            sleep(1)
            continue

        hwnd = windows[0]

        image = get_window_bitmap(hwnd)

        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        open_cv_image = cv2.resize(open_cv_image, (700, 400))
        cv2.imshow('frame', open_cv_image)
        a = cv2.waitKey(10)
        if a == 27:
            break

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
                    [perform_click(i, True) for i in
                     get_all_child_hwnd(hwnd) + [hwnd] if
                     'PowerPoint' in win32gui.GetWindowText(i)]
                    if videoEnabled:
                        t1 = time.monotonic()
                        signal_swipe = 'Назад'
                    t0 = False
                elif x_finger > x_board + w_board * 0.7:
                    [perform_click(i, False) for i in
                     get_all_child_hwnd(hwnd) + [hwnd] if
                     'PowerPoint' in win32gui.GetWindowText(i)]
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
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5)

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

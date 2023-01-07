import sys
import numpy as np
import cv2
import time
import mediapipe as mp

import pyautogui as pag
import threading

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class UiSettings(object):
    def setupUi(self, Settings):
        Settings.setObjectName("Settings")
        Settings.resize(500, 435)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(Settings.sizePolicy().hasHeightForWidth())
        Settings.setSizePolicy(sizePolicy)
        Settings.setMinimumSize(QtCore.QSize(500, 435))
        Settings.setMaximumSize(QtCore.QSize(500, 435))
        self.centralwidget = QtWidgets.QWidget(Settings)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(10, 0, 481, 41))
        font = QtGui.QFont()
        font.setFamily("Mongolian Baiti")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(60, 40, 381, 329))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.video = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.video.setObjectName("video")
        self.verticalLayout.addWidget(self.video)
        self.need_pointer = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.need_pointer.setObjectName("need_pointer")
        self.verticalLayout.addWidget(self.need_pointer)
        self.pointer_and_manager = QtWidgets.QComboBox(
            self.verticalLayoutWidget)
        self.pointer_and_manager.setEnabled(False)
        self.pointer_and_manager.setEditable(False)
        self.pointer_and_manager.setObjectName("pointer_and_manager")
        self.pointer_and_manager.addItem("")
        self.pointer_and_manager.addItem("")
        self.verticalLayout.addWidget(self.pointer_and_manager)
        self.commands = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.commands.setEnabled(False)
        self.commands.setObjectName("commands")
        self.commands.addItem("")
        self.commands.addItem("")
        self.verticalLayout.addWidget(self.commands)
        self.manager = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.manager.setEditable(False)
        self.manager.setObjectName("manager")
        self.manager.addItem("")
        self.manager.addItem("")
        self.verticalLayout.addWidget(self.manager)
        spacerItem = QtWidgets.QSpacerItem(20, 90,
                                           QtWidgets.QSizePolicy.Minimum,
                                           QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.apply_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.apply_btn.setObjectName("apply_btn")
        self.verticalLayout.addWidget(self.apply_btn)
        self.reference = QtWidgets.QPushButton(self.centralwidget)
        self.reference.setGeometry(QtCore.QRect(10, 370, 93, 28))
        self.reference.setObjectName("reference")
        Settings.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Settings)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 26))
        self.menubar.setObjectName("menubar")
        Settings.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Settings)
        self.statusbar.setObjectName("statusbar")
        Settings.setStatusBar(self.statusbar)

        self.retranslateUi(Settings)
        QtCore.QMetaObject.connectSlotsByName(Settings)

    def retranslateUi(self, Settings):
        _translate = QtCore.QCoreApplication.translate
        Settings.setWindowTitle(_translate("Settings", "Настройки"))
        self.label.setText(_translate("Settings", "Настройки"))
        self.video.setText(_translate("Settings", "Вывод видео"))
        self.need_pointer.setText(
            _translate("Settings", "Использование указки"))
        self.pointer_and_manager.setItemText(0, _translate("Settings",
                                                           "Правая рука для указки, левая для управления"))
        self.pointer_and_manager.setItemText(1, _translate("Settings",
                                                           "Левая рука для указки, правая для управления"))
        self.commands.setItemText(0, _translate("Settings", "ctrl+l"))
        self.commands.setItemText(1, _translate("Settings", "ctrl+ПКМ"))
        self.manager.setItemText(0, _translate("Settings",
                                               "Правая рука для управления"))
        self.manager.setItemText(1, _translate("Settings",
                                               "Левая рука для управления"))
        self.apply_btn.setText(_translate("Settings", "Применить"))
        self.reference.setText(_translate("Settings", "Справка"))


class UiReference(object):
    def setupUi(self, Reference):
        Reference.setObjectName("Reference")
        Reference.setEnabled(True)
        Reference.resize(412, 355)
        Reference.setMinimumSize(QtCore.QSize(412, 355))
        Reference.setMaximumSize(QtCore.QSize(412, 355))
        self.label = QtWidgets.QLabel(Reference)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(0, 0, 411, 41))
        font = QtGui.QFont()
        font.setFamily("Mongolian Baiti")
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.textBrowser = QtWidgets.QTextBrowser(Reference)
        self.textBrowser.setGeometry(QtCore.QRect(10, 40, 391, 301))
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(Reference)
        QtCore.QMetaObject.connectSlotsByName(Reference)

    def retranslateUi(self, Reference):
        _translate = QtCore.QCoreApplication.translate
        Reference.setWindowTitle(_translate("Reference", "Справка"))
        self.label.setText(_translate("Reference", "Справка"))
        self.textBrowser.setHtml(_translate("Reference",
                                            "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                            "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                            "p, li { white-space: pre-wrap; }\n"
                                            "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">1. &quot;Вывод видео&quot; - вывод видеопотока с камеры с ключевыми обозначениями (см. пункт 4).</span></p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">2. &quot;Использование указки&quot; - управление указкой с помощью указательного пальца выбранной руки (настройка в выпадающем списке ниже). Комбинация клавиш, которая включает указку в используемом приложении для презентации выбирается в выпадающем списке с клавишами). Для включения указки рукой для управления проведите от лица вверх.</span></p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">3. Если не включён пункт &quot;Использование указки&quot;, выберите руку, которая будет управлять презентацией (включение следующего или предыдущего слайда). Для листания слайдов проведите ладонью для управления от лица влево, чтобы переключиться на предыдущий слайд; от лица вправо, чтобы переключиться на следующий слайд.</span></p>\n"
                                            "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:8pt;\">4. Ключевые обозначения на видео - определение указательного пальца правой (белый поинт на пальце) или левой (жёлтый поинт на пальце) руки. При включённой указки выводится сообщение &quot;Указка включена&quot; на выводе потока видео, указательный палец руки, используемой для указки, выделяется красным поинтом. При переключении слайдов выводится сообщение &quot;Вперёд&quot; или &quot;Назад&quot; на выводе потока видео. Если рука для управления находится в меньшем прямоугольнике, то при перемещении ладони из этого прямоугольника в определённую сторону включится команда (см. пункт 3).</span></p></body></html>"))


class SettingsWidget(QMainWindow, UiSettings):
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


class ReferenceWidget(QMainWindow, UiReference):
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
    # Настраиваемые параметры ###
    videoEnabled = False  # вывод видео
    need_pointer = False  # использование указки
    hand_name_pointer, hand_name_manager = 'Right', 'Left'
    command = 'ctrl+l'
    # Конец настраиваемых параметров ###

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
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6)

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

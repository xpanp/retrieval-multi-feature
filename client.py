import sys
import cv2
import os 
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtGui, QtCore

from ui import Ui_retrieval
from search import search
from store import mysql

class RetrievalUI():
    def __init__(self, db_name) -> None:
        self.MainWindow = QMainWindow()
        self.ui_win = Ui_retrieval.Ui_MainWindow()
        self.ui_win.setupUi(self.MainWindow)
        self.ui_win.pushButton_choose.clicked.connect(self.openfile)
        self.ui_win.pushButton_search.clicked.connect(self.search)
        self.ui_win.comboBox.currentTextChanged.connect(self.mode_change)
        self.algorithm = self.ui_win.comboBox.currentText()
        print("current retrieval algorithm:", self.algorithm)

        self.img_path = " "
        self.search_engine = search.Search(mode = search.LOCAL, db = db_name)
        self.work_dir = os.getcwd()

    def show(self):
        self.MainWindow.show()

    # slot函数 选择文件
    def openfile(self):
        directory = QFileDialog.getOpenFileName(None,  "选取文件", self.work_dir, "All Files (*);")
        self.img_path = directory[0]
        if self.img_path == "":
            return
        img = cv2.imread(self.img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(rgb_img, rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[1]*3, QtGui.QImage.Format_RGB888)
        # TODO 不使用固定大小展示窗
        image = QtGui.QPixmap(image).scaled(100, 100, QtCore.Qt.KeepAspectRatio)
        self.ui_win.label.setPixmap(image)

    def label_show(self, scores, img_bufs):
        # TODO 展示四张图片的窗体不直接在ui中画死
        print("scores:", scores) # TODO 在窗体上直接展示
        self.ui_win.label_1.setPixmap(self.img_norm(img_bufs[0]))
        self.ui_win.label_2.setPixmap(self.img_norm(img_bufs[1]))
        self.ui_win.label_3.setPixmap(self.img_norm(img_bufs[2]))
        self.ui_win.label_4.setPixmap(self.img_norm(img_bufs[3]))

    # slot函数 检索
    def search(self):
        print("search picture path:", self.img_path)
        if len(self.img_path) == 0:
            print("please choose picture") # TODO 弹窗警告
            return
        scores, img_bufs = self.search_engine.search(self.img_path, self.algorithm)
        self.label_show(scores, img_bufs)

    # slot函数 检索方式改变
    def mode_change(self, algorithm):
        self.algorithm = algorithm
        print("retrieval algorithm change to:", self.algorithm)

    # 从图片路径获取qt label可展示的图片buffer
    def get_img(self, filepath):
        img = cv2.imread(filepath)
        img = self.img_norm(img)
        return img
    
    # 将cv2加载后的图片转换为qt label可展示的格式
    def img_norm(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(rgb_img, rgb_img.shape[1], rgb_img.shape[0], rgb_img.shape[1]*3, QtGui.QImage.Format_RGB888)
        # TODO 不使用固定大小展示窗
        img = QtGui.QPixmap(img).scaled(327, 197, QtCore.Qt.KeepAspectRatio)
        return img

if __name__ == '__main__':
    app = QApplication(sys.argv)
    r = RetrievalUI(mysql.TestDB)
    r.show()
    sys.exit(app.exec_())

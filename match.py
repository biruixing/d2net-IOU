import sys
#from PyQt5 import QFileDialog,QMessageBox,QApplication
#from PyQt5.QtGui import QImage,QPixmap
# from PyQt5.QtCore import *
# import PyQt5.sip
from PyQt5 import  QtWidgets,QtGui
from window import Ui_Dialog
# import os
# import cv2
import time
import glob
from d2net import *

class MyWindow(QtWidgets.QWidget,Ui_Dialog):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        # self.timer = QTimer(self)
        self.slot_init()
        # self.file_name=''
        # self.file_name2=''
        # self.img=None
        # self.rimgs_name=[]
        # self.img_left=None
        self.enable_match=False
        # self.args=None
        # self.device=None
        # self.model=None
        self.model,self.args,self.device=init_d2net()

    def slot_init(self):

        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.openfile2)
        self.pushButton_3.clicked.connect(self.do_match)
        # self.timer.timeout.connect(self.match_obj)
        # self.filename=''
        # self.filename2 = ''
        ##"open file Dialog "为文件对话框的标题，第三个是打开的默认路径，第四个是文件类型过滤器
    def openfile(self):
        # self.file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'open file dialog', '','img_file(*.tif)')
        self.file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'open file dialog', '', '*.jpg *.png *.jpeg *.tif;;*.*')
        self.lineEdit.setText(self.file_name[0])

        # qimg=QtGui.QImage(self.file_name[0])
        # if qimg.width()>self.width() or qimg.height()>self.height():
        #     qimg = QtGui.QImage
        img = cv2.imread(self.file_name[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,ch=img.shape
        self.keypoints1, self.scores1, self.descriptors1=extract_features(img,self.args,self.device,self.model)
        if h>self.label.height() or w>self.label.width():
            ratio = self.label.width() / max(h, w)
            img = cv2.resize(img,(int(w*ratio),int(h*ratio)))
            h, w, ch = img.shape
        qimg = QtGui.QImage(img.data,img.shape[1],img.shape[0],ch*w,QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))


    def openfile2(self):
        self.file_name2 = QtWidgets.QFileDialog.getExistingDirectory(self,
                  '选取文件夹',
                  './video')
        self.lineEdit_2.setText(self.file_name2)
        self.rimgs_name=glob.glob(self.file_name2+'/*.[jt][pi][gf]')
        # self.rimgs_name = glob.glob(self.file_name2 + '/*.tif')
        # self.rimgs_name=os.listdir(self.file_name2)self.file_name2+'/'+
        img2 = cv2.imread(self.rimgs_name[0])
        h,w,ch=img2.shape
        ratio=self.label.width()/max(h,w)
        if h>self.label.height() or w>self.label.width():
            img2 = cv2.resize(img2,(int(w*ratio),int(h*ratio)))
            h, w, ch = img2.shape
        qimg2 = QtGui.QImage(img2.data, img2.shape[1], img2.shape[0],ch*w, QtGui.QImage.Format_RGB888)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(qimg2))

    def do_match(self):
        if  self.file_name==''or self.file_name2=='':
            self.textBrowser.append('未选择图像路径')
            reply = QtWidgets.QMessageBox.warning(self,'warnning', '请选择图像路径')
            return

        #if self.timer.isActive():
        if self.pushButton_3.text()=='停止匹配':
            self.enable_match=False
            self.pushButton_3.setText('开始匹配')
            # self.timer.stop()
        else:
            self.enable_match=True

            img_left_large = cv2.imread(self.file_name[0],1)

            top, bottom = get_rect(img_left_large)
            # cv2.rectangle(img_left_large, top, bottom, (255, 0, 0), 2)

            is_reX=False
            is_reY=False
            xmin=top[1]
            xmax=bottom[1]
            ymin=top[0]
            ymax=bottom[0]
            # pt=sorted(self.keypoints[0,:,:])
            xc = (xmin+xmax)/2
            yc = (ymin+ymax)/2
            halfwc=(xmax-xmin)/2
            halfhc=(ymax-ymin)/2
            pt=[abs(self.keypoints1[:,0]-yc),abs(self.keypoints1[:,1]-xc)]
            pt2 = np.sort(pt)
            numrng=min(sum(pt2[0,:]<=halfhc),sum(pt2[1,:]<=halfwc))
            numThPt=1200
            if numrng<numThPt:
                m=len(pt[0])
                if m > numThPt:
                    ps = (pt2[0][numThPt],pt2[1][numThPt])
                else:
                    ps = (pt[0][m - 1],pt[1][m - 1])
                xmin=int(xc-ps[1])
                xmax=int(xc+ps[1])
                ymin=int(yc-ps[0])
                ymax=int(yc+ps[0])

            # for pp in pt2:
            #     if pp[0]<=hc and pp[1]<=wc:
            #         ccnt=ccnt+1
            #     else if ccnt<300:
            #         ccnt=ccnt+1
            # pt=self.keypoints[:,1,:]<xmax and self.keypoints[:,1,:]>xmin

            # self.img_left = img_left_large[top[1]:bottom[1], top[0]:bottom[0], :]
            self.img_left = img_left_large[xmin:xmax, ymin:ymax, :]
            keypoints_left, sorces_left, descriptor_left = extract_features(self.img_left, self.args, self.device, self.model)
            cv2.rectangle(img_left_large, top, bottom, (255, 0, 0), 2)
            cv2.putText(img_left_large,'box %d'%xmin+' %d'%ymin+' %d'%(xmax-xmin)+' %d'%(ymax-ymin),
                        (25,25),cv2.FONT_HERSHEY_PLAIN,0.8,(0,255,0))
            h, w, ch = img_left_large.shape
            ratio = self.label.width() / max(h, w)
            if h > self.label.height() or w > self.label.width():
                img_left_large = cv2.resize(img_left_large, (int(w * ratio), int(h * ratio)))
                h, w, ch = img_left_large.shape
            qimg = QtGui.QImage(img_left_large.data, img_left_large.shape[1], img_left_large.shape[0],ch*w, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

            self.pushButton_3.setText('停止匹配')
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)
            # while(self.enable_match):
            cnt=0
            model=None

            for rimg_path in self.rimgs_name:
                if self.enable_match==False:
                    break
                t1=cv2.getTickCount()
                # self.textBrowser.append('start match'+self.file_name[0]+ ' vs ' +self.file_name2+'/'+rimg_path)
                # img2,logs=main_d2(self.img_left,self.file_name2+'/'+rimg_path,is_reX,is_reY,cnt)
                self.textBrowser.append('start match '+self.file_name[0]+ ' vs ' +rimg_path)
                # model,img2,logs=main_d2(self.img_left,rimg_path,model,is_reX,is_reY,cnt)

                img2, logs = main_d2(self.img_left, rimg_path, self.model,self.args,self.device,
                                     keypoints_left, descriptor_left,sorces_left,
                                     is_reX, is_reY)
                # cv2.imwrite(rimg_path+'.jpg',img2)
                #img2, logs = main_d2(self.img_left, rimg_path, is_reX, is_reY, cnt)
                cnt=cnt+1
                # img2=cv2.resize(img2,(self.label_2.width(),self.height()))
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                h, w ,ch= img2.shape
                ratio=self.label.width()/max(h,w)
                if h > self.label.height() or w > self.label.width():
                    img2 = cv2.resize(img2, (int(w*ratio),int(h*ratio)))
                    h, w ,ch= img2.shape
                qimg2 = QtGui.QImage(img2.data, img2.shape[1], img2.shape[0],ch*w, QtGui.QImage.Format_RGB888)

                self.label_2.setPixmap(QtGui.QPixmap.fromImage(qimg2))
                for log in logs:
                    self.textBrowser.append(log)
                t2=cv2.getTickCount()
                print((t2-t1)/cv2.getTickFrequency())
                QtWidgets.QApplication.processEvents()
                time.sleep(0.01)
        self.enable_match = False
        self.pushButton_3.setText('开始匹配')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
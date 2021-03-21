import cv2
import sys
from PIL import Image
import time


def cp_wd(window_name,video_mode,picture_number,save_path):
    '''识别人脸并保存2000张人脸照片用于训练'''

    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)   #  给窗口命名并获取其模式
    #  窗口大小可以改变 cv2.WINDOW_NORMAL       窗口大小不可以改变 cv2.WINDOW_AUTOSIZE
    #  窗口大小自适应比例 cv2.WINDOW_FREERATIO  窗口大小保持比例  cv2.WINDOW_KEEPRATIO
    #  显示色彩变成暗色   cv2.WINDOW_GUI_EXPANDED

    video = cv2.VideoCapture(video_mode,cv2.CAP_DSHOW)
    #  获取视频来源，可输入视频路径，若将参数设为零，则为打开电脑摄像头，但要加上cv2.CAP_DSHOW

    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    #  使用人脸识别分类器，有不同类型，如haarcascade_lefteye_2splits.xml 表述检测左眼等等

    color = (60,250,200)      #  圈出人脸的边框颜色
    num = 0                   #  计数


    while (True):
        ret,frame = video.read()
        if ret == True:
            #  检验视频是否正常打开，是则读取其帧并赋予给frame
            grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            #  将刚刚获得的帧转变为灰度，这是为了减少计算

            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:          # 大于0则检测到人脸
                for faceRect in faceRects:  # 单独框出每一张人脸
                    x, y, w, h = faceRect   # x，y为图像矩阵左上角的坐标，w，h分别为宽和高
                    picture_name = '%s/%d.jpg'%(save_path,num)           #  保存照片的名字以及路径
                    #print(save_path)
                    picture = frame[y - 10: y + h + 10, x - 10: x + w + 10]     #  图片范围
                    cv2.imwrite(picture_name,picture)                           #  保存图片

                    num = num + 1
                    if num > picture_number:
                        break

                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                    #  第二和第三个参数为标记的位置，第四个为框的颜色，第五个为线条的宽度


                    #  显示拍了多少张照片
                    font = cv2.FONT_HERSHEY_SIMPLEX       #选择字体
                    cv2.putText(frame,'num:%d' % (num),(x-10 , y-10 ), font,2, (255,0,255),2)
                    #  各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细

            if num > picture_number:
                break
            #  达到数量后就直接跳出不需要等待后面输入q

            cv2.imshow(window_name,frame)
            #  显示图像

            c = cv2.waitKey(10)
            if c & 0xFF == ord('q'):
                break
            #  按q则结束
        else:
            break

    video.release()             #  释放摄像头
    cv2.destroyAllWindows()     #  销毁所有窗口

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        cp_wd("识别人脸区域", 0, 2000,'D:\\zyc_dm\\sy')





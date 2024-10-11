import cv2


img = cv2.imread('images/image.png')

classnames = []
classfile = 'files/coco.names'
with open(classfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
# print(classnames)

p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)   # الكشف والفحص
net.setInputSize(320, 230)           # تحديد حجم الصورة
net.setInputScale(1.0/127.5)         # القياس
net.setInputMean((127.5, 127.5, 127.5)) 
net.setInputSwapRB(True)               # نظام الألوان

classids, confs, bbox = net.detect(img, confThreshold=0.5)
# print(classids, bbox)

for classid, confidence, box in zip(classids.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, (0, 255, 0), 2)
    cv2.putText(img, classnames[classid-1],
                (box[0]+10, box[1]+20),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), thickness = 0)

cv2.imshow('program', img)
cv2.waitKey(0)
# save the image
cv2.imwrite('output/output_image4.png', img)

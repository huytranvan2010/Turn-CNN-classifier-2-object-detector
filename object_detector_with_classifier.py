from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from hammiu import pyramid, sliding_window
import numpy as np 
import argparse
import imutils
import time 
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)", help="ROI size in pixels")      # size of sliding window (width, height)
ap.add_argument("-c", "--min-conf", type=float, default=0.9, help="minimum probability to filter weak detection")
ap.add_argument("-v", "--visualize", type=int, default=-1, help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

# Một số biến cho object detection procedure
WIDTH = 600     # lấy width chung để resize vì các ảnh có size khác nhau
PYRAMID_SCALE = 1.5     # kích thước ảnh giảm đi sau mỗi lần thực hiện image pyramid
STEP_SIZE = 16   # khoảng cách sliding window dịch chuyển theo chiều ngang và dọc
ROI_SIZE =  eval(args["size"])  # lấy ROI_SIZE từ đó, do nó ở dạng string nên dùng eval để chuyển về tuple
INPUT_SIZE = (224, 244)     # input size phù hợp với mạng CNN sử dụng 

# tải pre-trained model weights
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=True)

# tải ảnh, resize lại theo WIDTH bên trên, lấy dimensions
original_image = cv2.imread(args["image"])
original_image = imutils.resize(original_image, width=WIDTH)
(H, W) = original_image.shape[:2]

# khởi tạo image pyramid (cái này là generator)
pyramid = pyramid(original_image, scale=PYRAMID_SCALE, minSize=ROI_SIZE)

# Tạo 2 list lưu ROIS được tạo ra từ image pyramid và sliding window (áp dụng cả 2 cái) 
# và lưu coordinates (x, y) - ví trí lấy ROIS để vẽ và putText
rois = []
locs = []

# Xác định time khi duyệt qua image pyramid và sliding windows
start = time.time()

# Duyệt qua image pyramid, đối với từng image trong pyramid mình sẽ thực hiện lấy các sliding window
for img in pyramid:
    # xác định scale factor (hệ số tỉ lệ) giữa ảnh "original" và lớp (ảnh) pyramid hiện tại (duyệt qua các lớp)
    # sử dụng để upscale bounding boxes tìm được ở các ảnh với size nhỏ hơn trong image pyramid cho phù hợp với size ảnh ban đầu
    scale = W / img.shape[1]

    # Ứng mỗi lớp (ảnh) của image pyramid duyệt qua các sliding windows
    for (x, y, roiOrig) in sliding_window(img, stepSize=STEP_SIZE, windowSize=ROI_SIZE):
        # scale lại (x, y) và kích thước cho phù hợp với original images (vì cái này trả cho image đã được downsample)
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        # nhớ những cái này phải chuyển về int do ở dạng pixel
        """ Chính vì có điều này, sliding window kích thước cố định nhưng mình có image pyramid do đó mình
        sẽ có rois với các kích thước khác nhau, không chỉ cố định theo size of sliding window"""

        # lấy vùng ROI (ko cần scale) vừa được lấy ra (từ các ảnh trong image pyramid), xử lý qua trước khi đưa vào CNN
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)     # convert to array
        roi = preprocess_input(roi)     # làm cho phù hợp trước khi đưa vào CNN, zero centering, chuyển về đúng color space...


        # Cập nhật list of ROIs và coordinates tương ứng
        rois.append(roi)
        locs.append((x, y, x + w, y + h))   # vị trí trên ảnh gốc

        # Check xem có biểu diễn sliding window ở image pyramid không
        if args["visualize"] > 0:
            # clone the original image và vẽ bounding boxes
            clone = original_image.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)    # vẽ các vị trí lấy ROIs

            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

end = time.time()
print("[INFO] Duyệt qua image pyramid và sliding windows mất: {:.5f} seconds".format(end - start))

# convert the ROIs to a numpy array
rois = np.array(rois, dtype="float32")

# Phân loại mỗi Rois sử dụng ResNet, xem mất bao nhiêu thời gian
start = time.time()
preds = model.predict(rois)     # dự đoán vật thể trong ROIs sử dụng ResNet50
end = time.time()
print("[INFO] classifying ROIs took {:.5} seconds".format(end - start))

# decode the predictions và khởi tạo dictionary để map class label đến ROIs liên hệ với label đó
preds = imagenet_utils.decode_predictions(preds, top=1)     # chỉ lấy top=1 với max proba, trả về list of list of tuple
labels = {}

# duyệt qua các predictions
for (i, p) in enumerate(preds):
    # lấy thông tin dự đoán cho mỗi ROI hiện tại
    (imagenetID, label, prob) = p[0]    # top=1 nên chỉ có 1 phần tử p[0]

    # lọc những dự đoán với probability lớn hơn mức yêu cầu
    if prob >= args["min_conf"]:
        # lấy bounding box liên hệ với dự đoán này
        box = locs[i]

        # lấy list của các dự đoán cho label và thêm bounding box + probabilityy vào list
        # nếu labels chưa có key label thì trả về empty list cho L, nếu có rồi thì trả về list có (box, prob) rồi thêm vào tiếp
        L = labels.get(label, [])   
        L.append((box, prob))
        labels[label] = L   # mỗi label có thể liên hệ với nhiều bounding boxes và probabilities tương ứng

# Sử dụng dictionary labels vừa tìm được đi annotate (chú thích các vật thể) trong ảnh
for label in labels.keys():
    # clone the original image so that we can draw on it
    print("[INFO] showing results for {}".format(label))
    clone = original_image.copy()

    for (box, prob) in labels[label]:
        # nhớ lại box có dạng (x, y, x+w, y+h)
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show kết quả trước khi áp dụng Non-max suppression
    cv2.imshow("Before", clone)
    
    # Áp dụng Non-max suppression
    clone = original_image.copy()

    boxes = np.array([p[0] for p in labels[label]])
    proba = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, proba)   # xem thêm package imutils

    """ Nên nhớ vẫn đang làm cho từng loại label """
    # Duyệt qua tất cả bounding boxes giữ lại sau non-max suppression
    for (startX, startY, endX, endY) in boxes:
        # draw the bounding and label on the image
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow("After", clone)
    cv2.waitKey(0)

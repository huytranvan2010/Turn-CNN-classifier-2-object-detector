import imutils

def sliding_window(image, stepSize, windowSize):    # windowSize = (width, height)
    """ Tạo các sliding windows """
    # trượt các cửa sổ theo ảnh
    for y in range(0, image.shape[0], stepSize):    # duyệt theo hàng -  height
        for x in range(0, image.shape[1], stepSize):    # duyệt theo cột - width
            # xuất ra từng cửa sổ
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])   # có cả vị trí cho window

def pyramid(image, scale=1.5, minSize=(30, 30)):    # minSize là (width, height)
    """ Tạo ra image pyramid """
    # xuất ra ảnh gốc
    yield image 

    while True:
        # tính size mới và resize ảnh
        w = int(image.shape[1] / scale)     # thay đổi width này
        image = imutils.resize(image, width=w)

        # nếu kích thước ảnh nhỏ hơn minimum size yêu cầu (theo bất cứ chiều nào) thì dừng, thoát luôn
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break 

        # xuất ra ảnh tiếp theo với size nhỏ hơn
        yield image
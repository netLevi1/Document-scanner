
import cv2 as cv
import numpy as np
import pyautogui


def resize(img):
    ScreenWidth, screenHeight = pyautogui.size()
    (imgHeight, imgWidth, _) = img.shape
    while True:
        if imgHeight < screenHeight and imgWidth < ScreenWidth:
            break
        imgWidth = int(imgWidth*0.95)
        imgHeight = int(imgHeight*0.95)

    resizedImg = cv.resize(img, (imgWidth, imgHeight))
    return resizedImg


def empty(x):
    pass


def blackWhite(img, blockSize=5, cparm=2):
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bwImg = cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, blockSize, cparm)
    return bwImg


def blackWhiteMain(img):
    cv.namedWindow("adjustments")
    cv.createTrackbar("blocksize", "adjustments", 3, 100, empty)
    cv.createTrackbar("C parameter", "adjustments", 0, 100, empty)
    while True:
        blockSize = cv.getTrackbarPos("blocksize", "adjustments")
        Cparameter = cv.getTrackbarPos("C parameter", "adjustments")
        if blockSize % 2 == 1 and blockSize > 2:  # blockSize must be odd
            img2 = blackWhite(img, blockSize, Cparameter)
        elif blockSize < 2:
            img2 = blackWhite(img, 3, Cparameter)
        else:
            img2 = blackWhite(img, blockSize+1, Cparameter)
        cv.imshow("uploaded image", img2)
        key = cv.waitKey(1)
        if key % 256 == 101:  # 'e' for exit
            cv.destroyWindow("adjustments")
            return img2


def grayScale(img):
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return grayImg


def drawFrame(img, imgName, msg):
    copyImg = img.copy()
    font = cv.FONT_HERSHEY_SIMPLEX
    textSize = cv.getTextSize(msg, font, 2, 3)[0]
    x_coor = int((copyImg.shape[1]-textSize[0])/2)
    y_coor = 50
    cv.putText(copyImg, msg, (x_coor, y_coor), font, 2, (0, 255, 0), 3)
    cv.imshow(imgName, copyImg)
    cv.waitKey(2000)


def fixWarp(img):
    cv.destroyWindow("uploaded image")
    # remove text from document
    originalImg = img.copy()
    kernel = np.ones((5, 5), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, 6)

    drawFrame(img, "remove text", "removing text")

    # use grabCut algorithm to detect document shape
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (20, 20, img.shape[1]-20, img.shape[0]-20)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]
    cv.destroyWindow("remove text")
    drawFrame(img, "detect shape", "shape detection")

    # canny edge detection
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayImg = cv.GaussianBlur(grayImg, (11, 11), 0)
    cannyImg = cv.Canny(grayImg, 0, 200)
    cannyImg = cv.dilate(
        cannyImg, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    con = np.zeros_like(img)
    # find contours and drawing them
    contours, hierarchy = cv.findContours(
        cannyImg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    doc = sorted(contours, key=cv.contourArea, reverse=True)[
        :5]  # take the biggest contours
    con = cv.drawContours(originalImg.copy(), doc, -1, (0, 255, 0), 3)

    cv.destroyWindow("detect shape")
    drawFrame(con, "contour detection", "contour detection")

    corners = findCorners(doc, img)
    corners = order_points(corners)
    destCorners = find_dest(corners)
    result = prespectiveTransform(corners, destCorners, originalImg)
    return result


def findCorners(doc, img):
    con = np.zeros_like(img)
    for x in doc:
        epsilon = 0.02 * cv.arcLength(x, True)
        corners = cv.approxPolyDP(x, epsilon, True)
  # If our approximated contour has four points
        if len(corners) == 4:
            break
    cv.drawContours(con, x, -1, (0, 255, 255), 3)
    cv.drawContours(con, corners, -1, (0, 255, 0), 10)
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    return corners


def order_points(pts):
    #  Rearrange coordinates to order:
    #  top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    arr = rect.astype('int').tolist()
    return arr


def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0],
                           [maxWidth, maxHeight], [0, maxHeight]]

    return order_points(destination_corners)


def prespectiveTransform(corners, destination_corners, orig_img):
    # Getting the homography.
    M = cv.getPerspectiveTransform(np.float32(
        corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv.warpPerspective(
        orig_img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv.INTER_LINEAR)
    return final


##########   MAIN PROGRAM   ############
print("Hello!\nWelcome to DocScanner.\nYou can either take a picture from a webcam or uploade one from local memory ")
userInput = input("Please choose your preferred way: 'webcam' or 'upload': ")
fixedMode = False
isFixed = False  # indicates weather the image has been fixed with Perspective Transformation
blackAndWhiteMode = False
fixedImg = None
imgCounter = 0

if userInput == 'upload':
    filePath = input("enter file path: ")
    print("************************************\nPress SPACE to save a photo\nPress 'f' to fix image prespective\nPress 'g' for grayscale image\nPress 'b' for Black & White image\nPress 'e' to exit Black & White mode\nPress 'o' for original image\nPress ESC to exit\n************************************")
    docImg = cv.imread(filePath, cv.IMREAD_COLOR)
    resizedImg = resize(docImg)
    cv.imshow("uploaded image", resizedImg)
    key = cv.waitKey(0)
    originalImg = resizedImg.copy()
    while True:
        if key % 256 == 27:  # ESC to exit
            break
        # if x was pressed close the program
        elif cv.getWindowProperty("uploaded image", cv.WND_PROP_VISIBLE) < 1:
            break
        elif key % 256 == 103:  # 'g' for greyscale image
            if not fixedMode:
                updatedImage = grayScale(originalImg)
            else:
                updatedImage = grayScale(fixedImg)
        elif key % 256 == 111:  # 'o' for original image
            fixedMode = False
            updatedImage = originalImg
        elif key % 256 == 102:  # 'f' to fix image warping if needed
            if not isFixed:
                fixedMode = True
                isFixed = True
                updatedImage = fixWarp(originalImg)
                cv.destroyWindow("contour detection")
                fixedImg = updatedImage
            else:
                updatedImage = fixedImg
        elif key % 256 == 98:  # 'b' for Black&White image
            if not fixedMode:
                updatedImage = blackWhiteMain(originalImg)
            else:
                updatedImage = blackWhiteMain(fixedImg)
        elif key % 256 == 99:  # 'c' for colored image
            if not fixedMode:
                updatedImage = originalImg
            else:
                updatedImage = fixedImg
        elif key % 256 == 32:  # SPACE to take a photo and save it to current folder
            imgName = "imgNumber"+str(imgCounter)
            cv.imwrite(imgName+".png", updatedImage)
            print("photo saved")
            imgCounter = imgCounter+1
          # update image accorging to input:
        else:
            print("key is not supported")
        cv.imshow("uploaded image", updatedImage)
        key = cv.waitKey(0)

    cv.destroyAllWindows()

elif userInput == 'webcam':
    s = 0  # default webcam
    source = cv.VideoCapture(s)
    cv.namedWindow("Camera preview", cv.WINDOW_KEEPRATIO)
    print("************************************\nPress SPACE to save a photo\nPress ESC to exit\n************************************")
    while True:
        has_frame, frame = source.read()
        if not has_frame:
            break
        key = cv.waitKey(1)
        if key % 256 == 27:  # ESC to exit program
            break
        elif key % 256 == 32:  # SPACE to take a photo and save it to current folder
            imgName = "imgNumber"+str(imgCounter)
            cv.imwrite(imgName+".png", frame)
            print("photo taken")
            imgCounter = imgCounter+1
        # if x was pressed close the program
        elif cv.getWindowProperty("Camera preview", cv.WND_PROP_VISIBLE) < 1:
            break
        if key % 256 != 32 and key % 256 != 27 and key != -1:
            print("key is not supported")
        cv.imshow("Camera preview", frame)
    source.release()
    cv.destroyAllWindows()

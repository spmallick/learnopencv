import cv2
import sys
import numpy as np


def displayBboxOpenCV(im, bbox):
	if bbox is not None:
		bbox = bbox.astype(int)
		n = len(bbox[0])
		for i in range(n):
			cv2.line(im, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % n]), yellow, 3)


def displayBboxWeChat(im, bbox):
	if bbox is not None:
		bbox = [bbox[0].astype(int)]
		n = len(bbox[0])
		for i in range(n):
			cv2.line(im, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % n]), green, 3)


def opencvQR(im, qrDecoder):
	# Detect and decode.
	data, bbox, rectifiedImg = qrDecoder.detectAndDecode(im)
	if len(data) > 0:
		cv2.putText(im, 'OpenCV Output: {}'.format(data), (20, im.shape[0] - 25), font, fontScale, yellow, 2)
		displayBboxOpenCV(im, bbox)
		print('QR Data [ OpenCV ]: ', data)
	else:
		print('QR Code not detected by OpenCV')
		cv2.putText(im, 'OpenCV Output: Not Detected', (20, im.shape[0] - 25), font, fontScale, red, 2)
	return im


def wechatQR(im, detector):
	# Detect and decode.
	res, points = detector.detectAndDecode(im)
	if len(res) > 0:
		print('QR Data [ Wechat ]: ', res)
		cv2.putText(im, 'WeChat Output: {}'.format(res[0]), (20, im.shape[0] - 25), font, fontScale, green, 2)
		displayBboxWeChat(im, points)
	else:
		print('QRCode not detected by WeChat')
		cv2.putText(im, 'WeChat Output: Not Detected', (20, im.shape[0] - 25), font, fontScale, red, 2)
	return im


#==============================================CONSTANTS================================================#
# Color.
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
yellow = (0,255,255)
# Font.
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.8

#============================================INITIALIZATIONS==============================================#
# Instantiate OpenCV QR Code detector.
qrDecoder = cv2.QRCodeDetector()

# Instantiate WeChat QR Code detector.
weChatDetector = cv2.wechat_qrcode_WeChatQRCode("../model/detect.prototxt",
	"../model/detect.caffemodel",
	"../model/sr.prototxt",
	"../model/sr.caffemodel")
#=========================================================================================================#


if __name__== '__main__':

	if len(sys.argv)>1:
		vidCapture = cv2.VideoCapture(sys.argv[1])
	else:
		vidCapture = cv2.VideoCapture(0)

	frameWidth = int(vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frameHeight = int(vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))

	output = cv2.VideoWriter('comparison-video.mp4', 
		cv2.VideoWriter_fourcc(*'XVID'), 30, (2*frameWidth, frameHeight))

	while (vidCapture.isOpened()):
		ret, frame = vidCapture.read()
		if not ret:
			print('Error reading frames.')
			break
		img = frame.copy()
		# Call OpenCV QR Code scanner.
		outOpenCV = opencvQR(img.copy(), qrDecoder)
		# Call WeChat QR Code scanner.
		outWeChat = wechatQR(img.copy(), weChatDetector)
		# Concatenate outputs.
		result = cv2.hconcat([outOpenCV, outWeChat])
		output.write(result)
		cv2.imshow('Frame',result)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break

	output.release()
	vidCapture.release()
	cv2.destroyAllWindows()

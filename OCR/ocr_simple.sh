# Usage sh ocr_simple.sh image.jpg

# '${1}' passes image.jpg as an input
# 'stdout' ensures that result is printed on terminal
# If some other name is specified say output, tesseract
# will write result in file named output.txt
# '-l eng'  for using the English language
# '--oem 1' sets the OCR Engine Mode to LSTM only
# '--psm 3' sets the Page Segmentation Mode (psm) to auto

#  There are four OCR Engine Mode (oem) available
#  0    Legacy engine only.
#  1    Neural nets LSTM engine only.
#  2    Legacy + LSTM engines.
#  3    Default, based on what is available.
#
#  '--psm 3' sets the Page Segmentation Mode (psm) to auto.
#  Other important psm modes will be discussed in a future post.

tesseract ${1} stdout -l eng --oem 1 --psm 3


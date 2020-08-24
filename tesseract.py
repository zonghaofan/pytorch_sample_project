from PIL import Image
import pytesseract
import time

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
st_time = time.time()
# for i in range(20):
print(pytesseract.image_to_string(Image.open('8.png')))
print('平均时间{}'.format((time.time() - st_time)/100))
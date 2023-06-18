# from paddleocr import PaddleOCR,draw_ocr
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
import streamlit as st
import re
import csv
import pdb
from dotenv import load_dotenv
load_dotenv()
# Specify the directories for input and output files
image_path = os.getenv('IMAGE_PATH')
csv_ = os.getenv('CSV_')
# det_img_path = '/home/hien/Documents/KIE/KIE_invoice/data/result/det_img/'
result_cropped = os.getenv('RESULT_CROPPED')

def ocr(img):
    # img = 'ahah.jpg' -> file_name = ahah
    file_name = re.sub(r'\.png|\.jpg|\.jpeg', '', img)
    file_ocr = file_name + '.csv'
    # Initialize the PaddleOCR instance
    ocr = PaddleOCR(det_model_dir='./weights/text_recog/east_r50_vd/', 
                    rec_model_dir='./weights/text_recog/r34_vd_tps_bilstm_ctc/',
                    )
    # Perform OCR on the image
    result = ocr.ocr(image_path + img)
    # Draw the result on the image and save it
    result = result[0]
    # image = Image.open(image_path + img).convert('RGB')
    with open(csv_ + file_ocr, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["xmin", "ymin", "xmax", "ymax", "object", "label"])
        for line in result:
            ls_x = []
            ls_y = []
            boxes_t = ''
            for i in line[0]:
                ls_x.append(int(i[0]))
                ls_y.append(int(i[1]))
            xmin = min(ls_x)
            ymin = min(ls_y)
            xmax = max(ls_x)
            ymax = max(ls_y)
            boxes_t = boxes_t + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax)
            txts_t = line[1][0]
            if ',' in txts_t:
                txts_t = '"' + txts_t + '"'
            file.write(boxes_t + "," + txts_t + '\n')
            # pdb.set_trace()
    return csv_ + file_ocr

img = '321.jpg'
# Perform OCR on the images
ls_re = ocr(img)




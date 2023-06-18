import os
import glob
import time
import cv2
import torch
import numpy as np
from PIL import Image
from kie.model import InvoiceGCN
from kie.dataset import make_graph
from processing import norm_date
from processing import norm_invoice
import matplotlib.pyplot as plt
import pdb
from ocr import ocr
import streamlit as st
image_path = os.getenv('IMAGE_PATH')
image_path_kie = os.getenv('IMAGE_PATH_KIE')
test_output_fd = "/home/hien/Documents/KIE/KIE_invoice/data/result/test_output_fd"
if not os.path.exists(test_output_fd):
    os.mkdir(test_output_fd)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
col1, col2, col3 = st.columns(3)
st.title("DATN DO VU THANH HIEN 20182494")
st.header("Key information extraction from invoice")
with st.form("form1", clear_on_submit=True):
            content_file = st.file_uploader(
                "Upload your image here", type=["jpg", "jpeg", "png"]
            )
            submit = st.form_submit_button("Upload")
            if content_file is not None:
                ocr_img = content_file.name
                pil_img = Image.open(content_file)
                img = np.array(pil_img)
                imagecopy= np.copy(img)
                if submit:
                    print(">" * 100)
                    wait_text = st.text("Please wait...")
                    res_ocr = ocr(ocr_img)
                    
                    train_data = torch.load(os.path.join("/home/hien/Documents/KIE/KIE_invoice/model/GCN_data/processed", 'train_data.dataset'))

                    model = InvoiceGCN(input_dim=train_data.x.shape[1], chebnet=True)
                    checkpoint = torch.load('/home/hien/Documents/KIE/KIE_invoice/weights/kie/best.pt', map_location=device)
                    model.load_state_dict(checkpoint)
                    model.eval()

                    classes = ["company", "address", "invoice", "date","total","other"]

                    # files = glob.glob("/home/hien/Documents/KIE/KIE_invoice/data/result/csv_/*.csv")

                    print("path: ", res_ocr)

                    # img = cv2.imread(image_path + ocr_img)
                    # pdb.set_trace()
                    cv2.imwrite(image_path_kie + ocr_img, img)
                    t0 = time.time()
                    # pdb.set_trace()
                    individual_data, df = make_graph(res_ocr, classes)
                    y_pred = model(individual_data).max(dim=1)[1]
                    res_txt_kie = []
                    for row_index, row in df.iterrows():
                        x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']].astype("int")
                        true_label = row["label"]
                        object_ = row['object']
                        # if isinstance(true_label, str) and true_label != "other":
                        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        _y_pred = y_pred[row_index]
                        if _y_pred != len(classes)-1:
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                            _label = classes[_y_pred]
                            if _label == classes[3]:
                                 object_ = norm_date(object_)
                            if _label == classes[2]:
                                 object_ = norm_invoice(object_)
                            cv2.putText(
                                img, "{}".format(_label), (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                            )
                            res_txt_kie.append(_label + " " + object_)
                            print("=====", _label + " " + object_)
                            

                    print("Time process: ", time.time()-t0)

                    if(img.shape[0] < 2000):
                        scale_percent = 100 # percent of original size
                    else:
                        scale_percent = 50
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    # resize image
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    
                    resized_img_path = os.path.join(test_output_fd, '{}_result.jpg'.format((os.path.basename(image_path + ocr_img)).split('.')[0]))
                    cv2.imwrite(resized_img_path, resized)
                    wait_text.empty()
                    with col1:
                        st.image(imagecopy)
                    with col2:
                        st.image(resized)
                    with col3:
                        for i in res_txt_kie:
                             text_write = i
                             st.text(text_write)
                        total_runtime = round(float(time.time()-t0), 4)
                        st.text("Total runtime: {}s".format(total_runtime))
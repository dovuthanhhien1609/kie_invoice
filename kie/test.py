import os
import glob
import time
import cv2
import torch
from KIE_invoice.model.kie.model import InvoiceGCN
from KIE_invoice.model.kie.dataset import make_graph
import matplotlib.pyplot as plt
import pdb
test_output_fd = "./test_output"
if not os.path.exists(test_output_fd):
    os.mkdir(test_output_fd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = torch.load(os.path.join("./GCN_data/processed", 'train_data.dataset'))

model = InvoiceGCN(input_dim=train_data.x.shape[1], chebnet=True)
checkpoint = torch.load('./weights/best.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()

classes = ["company", "address", "invoice", "date","total","other"]

files = glob.glob("./GCN_data/csv/*.csv")

for file in files:
    print("path: ", file)
    image_path = file.replace(".csv", ".jpg").replace("/csv/", "/images/")
    img = cv2.imread(image_path)
    
    t0 = time.time()
    individual_data, df = make_graph(file, classes)
    y_pred = model(individual_data).max(dim=1)[1]
 
    for row_index, row in df.iterrows():
        x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']].astype("int")
        true_label = row["label"]
        # pdb.set_trace()
        object_ = row['object']

        # if isinstance(true_label, str) and true_label != "other":
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _y_pred = y_pred[row_index]
        if _y_pred != len(classes)-1:
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            _label = classes[_y_pred]
            cv2.putText(
                img, "{}".format(_label), (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            print("=====", object_ + ":" + _label)
            # pdb.set_trace()

    print("Time process: ", time.time()-t0)

    if(img.shape[0] < 2000):
        scale_percent = 80 # percent of original size
    else:
        scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("img", resized)
    cv2.waitKey()
    # plt.savefig(os.path.join(test_output_fd, '{}_result.png'.format(os.path.basename(image_path))), bbox_inches='tight')

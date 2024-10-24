from celery_tasks.crnn import CRNN_Model

import pandas as pd


# GPU Mapping
#Code:Machine
#1:3
#0:1
#2:0
#3:2

crnn_model = CRNN_Model(gpu_id="0")
img_folder = 'data/test/test_imgs/test/'

img_list = pd.read_csv('data/test/written_name_test_v2.csv')

data = {"img_list":img_list,
        "img_folder":img_folder,
        "n_imgs":2000,
        }


predicted_words = crnn_model.words_predict_dev(data["img_list"], data["img_folder"], data["n_imgs"])

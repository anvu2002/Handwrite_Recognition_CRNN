from celery_tasks.crnn import CRNN_Model
import pandas as pd



crnn_model = CRNN_Model()
img_folder = 'data/test/test_imgs/test/'

img_list = pd.read_csv('data/test/written_name_test_v2.csv')

# data = {"img_list":img_list,
#         "img_folder":img_folder,
#         "n_imgs":10,
#         }

data = {    
        "img_path":"api/uploads/TEST_0066.jpg"
}

# predicted_words = crnn_model.words_predict(data["img_list"], data["img_folder"], data["n_imgs"])

predicted_words = crnn_model.words_predict(data["img_path"])

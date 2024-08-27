from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode, get_value
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger



def preprocess_img(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
def num_to_label(num):
    alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "

    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

def preprocess_input(img_list, n_img):
    img_set = []

    plt.figure(figsize=(15, 10))
    for i in range(n_img):
        # ax = plt.subplot(2, 3, i+1)
        img_dir = './data/test/test_v2/test_imgs/'+img_list.loc[i, 'FILENAME']

        image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        plt.imshow(image, cmap='gray')

        image = preprocess_img(image)
        image = image/255.
        img_set.append(image)

    return img_set
def prepare_prediction(pred, n_img):
    # Decode the predictions
    decoded = ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0]
    # Convert the decoded predictions to numpy array
    decoded = get_value(decoded)

    res = [ num_to_label(decoded[i]) for i in range(n_img)]
    return res

def evaluate(img_list, n_img, predicted_result):
    y_true = img_list.loc[0:n_img, 'IDENTITY']
    correct_char = 0
    total_char = 0
    correct = 0

    true_result = []

    for i in range(n_img):
        pr = predicted_result[i]
        tr = y_true[i]
        total_char += len(tr)
        
        for j in range(min(len(tr), len(pr))):
            if tr[j] == pr[j]:
                correct_char += 1
                
        if pr == tr :
            correct += 1 
        
        true_result.append(tr)
    
    logger.info('Correct characters predicted : %.2f%%' %(correct_char*100/total_char))
    logger.info('Correct words predicted      : %.2f%%' %(correct*100/n_img))
    logger.info(f'\nTrue Results = {true_result}')


model = load_model("./celery_tasks/trained_models/CRNN_Handwrite_model.keras")
img_list = pd.read_csv('./data/test/test_v2/written_name_test_v2.csv')

input_set = preprocess_input(img_list, 10)

pred = model.predict(np.array(input_set).reshape(-1, 256, 64, 1))
   
result = prepare_prediction(pred,10)

   
logger.info(f"Predict Words = {result}")

evaluate(img_list, 10, result)


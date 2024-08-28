from loguru import logger
from celery import Task
from celery.exceptions import MaxRetriesExceededError
from .app_worker import app
# from .yolo import YoloModel
from .crnn import CRNN_Model



# class PredictObjectTask(Task):
#     def __init__(self):
#         super().__init__()
#         self.model = None

#     def __call__(self, *args, **kwargs):
#         if not self.model:
#             logger.info('Loading YOLO Model...')
#             self.model = YoloModel()
#             logger.info('[+] YOLO Model loaded')
#         return self.run(*args, **kwargs)

class PredictWordsTask(Task):
    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        if not self.model:
            logger.info('Loading CRNN Model...')
            self.model = CRNN_Model()
            logger.info('[+] CRNN Model loaded')
        return self.run(*args, **kwargs)


# @app.task(ignore_result=False, bind=True, base=PredictObjectTask)
# def predict_image(self, data):
#     try:
#         data_pred = self.model.predict(data)
#         return {'status': 'SUCCESS', 'result': data_pred}
#     except Exception as ex:
#         try:
#             self.retry(countdown=2)
#         except MaxRetriesExceededError as ex:
#             return {'status': 'FAIL', 'result': 'max retried achieved'}

@app.task(ignore_result=False, bind=True, base=PredictWordsTask)
def dectect_words(self, data):
    try:
        predicted_words = self.model.words_predict(data["img_path"])
        return {'status': 'SUCCESS', 'result': predicted_words}
    except Exception as ex:
        try:
            self.retry(countdown=2)
        except MaxRetriesExceededError as ex:
            return {'status': 'FAIL', 'result': 'max retried achieved'}
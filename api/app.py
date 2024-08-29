import sys
import os
sys.path.insert(0, os.path.realpath(os.path.pardir))
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# from celery_tasks.tasks import predict_image
from celery_tasks.tasks import dectect_words
from celery.result import AsyncResult
from models import Task, Prediction
import uuid
import logging
# from pydantic.typing import List
from loguru import logger

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/results'

isdir = os.path.isdir(UPLOAD_FOLDER)
if not isdir:
    os.makedirs(UPLOAD_FOLDER)

isdir = os.path.isdir(STATIC_FOLDER)
if not isdir:
    os.makedirs(STATIC_FOLDER)

origins = ["*"]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post('/api/process')
# async def process(files: list[UploadFile] = File(...)):
#     tasks = []
#     try:
#         for file in files:
#             d = {}
#             try:
#                 name = str(uuid.uuid4()).split('-')[0]
#                 ext = file.filename.split('.')[-1]
#                 file_name = f'{UPLOAD_FOLDER}/{name}.{ext}'
#                 with open(file_name, 'wb+') as f:
#                     f.write(file.file.read())
#                 f.close()

#                 # start task prediction
#                 task_id = predict_image.delay(os.path.join('api', file_name))
#                 d['task_id'] = str(task_id)
#                 d['status'] = 'PROCESSING'
#                 d['url_result'] = f'/api/result/{task_id}'
#             except Exception as ex:
#                 logging.info(ex)
#                 d['task_id'] = str(task_id)
#                 d['status'] = 'ERROR'
#                 d['url_result'] = ''
#             tasks.append(d)
#         return JSONResponse(status_code=202, content=tasks)
#     except Exception as ex:
#         logging.info(ex)
#         return JSONResponse(status_code=400, content=[])


@app.post('/api/detect_words')
async def process(files: list[UploadFile] = File(...)):
    tasks = []
    try:
        for file in files:
            d = {}
            try:
                # save uploaded files
                try:
                    contents = file.file.read()
                    file_name = f'{UPLOAD_FOLDER}/{file.filename}'
                    with open(file_name, 'wb') as f:
                        f.write(contents)
                except Exception:
                    return {"message": "There was an error uploading the file(s)"}
                finally:
                    file.file.close()                
                

                logger.debug(f"Saved image -- {os.path.join('api', file_name)}")
                logger.debug("Requesting CRNN ...")


                # start word prediction
                data = {}
                data.update({"img_path":os.path.join('api', file_name)})
                

                task_id = dectect_words.delay(data)
                logger.debug(f"task_id = {task_id}")
                d['task_id'] = str(task_id)
                d['status'] = 'PROCESSING'
                d['url_result'] = f'/api/result/{task_id}'
            except Exception as ex:
                logging.info(ex)
                d['task_id'] = str(task_id)
                d['status'] = 'ERROR'
                d['url_result'] = ''
            tasks.append(d)

        return JSONResponse(status_code=202, content=tasks)
    except Exception as ex:
        logger.debug(f"PROCESSING ERROR: {ex}")
        return JSONResponse(status_code=500, content=[])



@app.get('/api/result/{task_id}', response_model=Prediction)
async def result(task_id: str):
    task = AsyncResult(task_id)

    # Task Not Ready
    if not task.ready():
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': task.status, 'result': ''})

    # Task done: return the value
    task_result = task.get()
    result = task_result.get('result')
    return JSONResponse(status_code=200, content={'task_id': str(task_id), 'status': task_result.get('status'), 'result': result})


@app.get('/api/status/{task_id}', response_model=Prediction)
async def status(task_id: str):
    task = AsyncResult(task_id)
    return JSONResponse(status_code=200, content={'task_id': str(task_id), 'status': task.status, 'result': ''})

@app.get('/api/health')
async def health():
    return JSONResponse(status_code=200, content={'server':'crnn_worker','status':'HEALTHY'})

@app.post("/files")
async def UploadImage(file: bytes = File(...)):
    with open('image.jpg','wb') as image:
        image.write(file)
        image.close()
    return 'got it'
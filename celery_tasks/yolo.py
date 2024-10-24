from loguru import logger
import torch
import matplotlib.pyplot as plt


class YoloModel:
    def __init__(self, gpu_id: str = "1", memory_limit: int = None):
        # Set the specified GPU ID
        self.gpu_id = gpu_id
        self.memory_limit = memory_limit
        self._set_device()

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.model.eval()
    
    def _set_device(self):
        # Set the CUDA device
        if torch.cuda.is_available():
            # Set the GPU device based on the specified ID
            device = torch.device(f'cuda:{self.gpu_id}')
            logger.debug(f"[YOLO] Using GPU: {self.gpu_id}")

            # Limit memory usage if specified
            if self.memory_limit:
                torch.cuda.set_per_process_memory_fraction(self.memory_limit / 1024 / 1024, device=device)
        else:
            device = torch.device('cpu')
            logger.warning("CUDA is not available. Using CPU.")

        # Move the model to the appropriate device
        self.model.to(device)
        
        logger.info("[YOLO] Model loaded on device: {}".format(device))

    def predict(self, img):
        try:
            with torch.no_grad():
                result = self.model(img)
            result.save('api/static/results/')
            final_result = {}
            data = []
            file_name = f'static/{result.files[0]}'

            for i in range(len(result.xywhn[0])):
                x, y, w, h, prob, cls = result.xywhn[0][i].numpy()
                preds = {}
                preds['x'] = str(x)
                preds['y'] = str(y)
                preds['w'] = str(w)
                preds['h'] = str(h)
                preds['prob'] = str(prob)
                preds['class'] = result.names[int(cls)]
                data.append(preds)

            return {'file_name': file_name, 'bbox': data}
        except Exception as ex:
            logging.error(str(ex))
            return None

# docker run -it --rm --name test-jupyter --gpus all -v .:/app --runtime=nvidia tensorflow/tensorflow:latest-gpu

# docker run -it --rm \
#     --name test-jupyter \
#     --gpus all \
#     -v .:/app \
#     --runtime=nvidia \
#     tensorflow/tensorflow:latest-gpu \
#     # bash -c "cd /app && exec bash"


docker run -it --rm \
    --name crnn-worker \
    --gpus all \
    -p 7777:7777 \
    crnn-image


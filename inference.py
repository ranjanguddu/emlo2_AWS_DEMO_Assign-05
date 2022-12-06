import torch
import gradio as gr
import torchvision.transforms as T
import boto3
import botocore
import traceback

print("Downloading model from S3")
BUCKET_NAME = "guddubucket"
KEY = "model.script.pt"
# file = "model.script.pt"
# s3 = boto3.resource('s3')

# try:
#     s3.Bucket(BUCKET_NAME).download_file(KEY, file)
# except botocore.exceptions.ClientError as e:
#     if e.response['Error']['Code'] == "404":
#         print("The object does not exist.")
#     else:
#         raise
# print("Model downloaded, and saved ")

s3 = boto3.client('s3')
s3.download_file(BUCKET_NAME, 'EMLO_ASSIGN_05/model.script.pt', 'model1.script.pt')

model = torch.jit.load('model1.script.pt')
model.eval()

# log.info(f"Loaded Model: {model}")

def recognize_image(image):
    if image is None:
        return None
    image = T.ToTensor()(image).unsqueeze(0)
    try:
        preds = model.forward_jit(image)
    except Exception:
        traceback.print_exc()
    preds = preds[0].tolist()
    label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return {label[i]: preds[i] for i in range(10)}


im = gr.Image(shape=(32, 32))
demo = gr.Interface(
    fn=recognize_image,
    inputs=[im],
    outputs=[gr.Label(num_top_classes=10)]

)
demo.launch(server_name="0.0.0.0", server_port=80)
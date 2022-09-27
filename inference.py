import uvicorn
import cv2
import numpy as np
import time
import argparse
import torch
import os
import sys
from io import StringIO 
import io
import base64
from PIL import Image
from fastapi.responses import FileResponse
sys.path.append('./decalib/datasets')

from scipy.io import savemat
from tqdm import tqdm
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.utils.preprocess import preProcessImage
from pydantic import BaseModel

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_methods = ['*'],
    allow_headers = ['*']
)

class Item(BaseModel):
    dataUrl: str
    bbox: list

@app.get("/")
async def root():
    return {"Test succesful"}

# Defining the default parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
iscrop = True
sample_step = 10
detector = 'fan'
rasterizer_type = 'standard'
render_orig = True
useTex = True
extractTex = True
saveVis = True
saveKpt = False
saveDepth = False
saveObj = True
saveMat = False
saveImages = False

deca_cfg.model.use_tex = useTex
deca_cfg.rasterizer_type = rasterizer_type
deca_cfg.model.extract_tex = extractTex
deca = DECA(config = deca_cfg, device=device)


@app.post('/get3DFaceAvatar')
async def get3DFaceAvatar(file: UploadFile):

    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("Virtual User Face Avatar Data Received")
    # print(item)
    # image_data_url = item.dataUrl
    # sbuf = StringIO()
    # sbuf.write(image_data_url)
    # b = io.BytesIO(base64.b64decode(image_data_url))
    # pimg = Image.open(b)
    image = np.array(frame)

    # Processing the image
    testData = preProcessImage(image, [])
    images = testData['image'].to(device)[None,...]
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict) #tensor
        if render_orig:
            tform = testData['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testData['original_image'][None, ...].to(device)
            _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
            orig_visdict['inputs'] = original_image
    deca.save_obj(os.path.join('.', 'temp.obj'), opdict)
    with open("temp.obj", "rb") as file:
        faceModel = base64.b64encode(file.read())
    with open("temp.mtl", "rb") as file:
        faceMaterial = base64.b64encode(file.read())
    with open("temp.png", "rb") as file:
        faceTexture = base64.b64encode(file.read())
    with open("temp_normals.png", "rb") as file:
        faceNormal = base64.b64encode(file.read())

    return {
        "faceModel" : faceModel,
        "faceTexture" : faceTexture,
        "faceMaterial" : faceMaterial,
        "faceNormal": faceNormal
    }

@app.post('/get3DFaceAvatarfromURL')
async def get3DFaceAvatarfromURL(item: Item):

    print("Physical User Face Avatar Data Received")
    image_data_url = item.dataUrl
    sbuf = StringIO()
    sbuf.write(image_data_url)
    b = io.BytesIO(base64.b64decode(image_data_url))
    pimg = Image.open(b)
    image = np.array(pimg)

    # Processing the image
    testData = preProcessImage(image, item.bbox, original = True)
    images = testData['image'].to(device)[None,...]
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict) #tensor
        if render_orig:
            tform = testData['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testData['original_image'][None, ...].to(device)
            _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
            orig_visdict['inputs'] = original_image
    deca.save_obj(os.path.join('.', 'temp.obj'), opdict)
    with open("temp.obj", "rb") as file:
        faceModel = base64.b64encode(file.read())
    with open("temp.mtl", "rb") as file:
        faceMaterial = base64.b64encode(file.read())
    with open("temp.png", "rb") as file:
        faceTexture = base64.b64encode(file.read())
    with open("temp_normals.png", "rb") as file:
        faceNormal = base64.b64encode(file.read())

    return {
        "faceModel" : faceModel,
        "faceTexture" : faceTexture,
        "faceMaterial" : faceMaterial,
        "faceNormal": faceNormal
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
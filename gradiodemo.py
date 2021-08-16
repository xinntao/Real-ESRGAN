import os
import random
import gradio as gr
from PIL import Image
import torch
from random import randint
import sys
from subprocess import call
    
  
    
def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)
run_cmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P .")
# run_cmd("pip freeze")
run_cmd("python setup.py develop")
def inference(img):
    _id = randint(1, 10000)
    INPUT_DIR = "/tmp/input_image" + str(_id) + "/"
    OUTPUT_DIR = "/tmp/output_image" + str(_id) + "/"
    run_cmd("rm -rf " + INPUT_DIR)
    run_cmd("rm -rf " + OUTPUT_DIR)
    run_cmd("mkdir " + INPUT_DIR)
    run_cmd("mkdir " + OUTPUT_DIR)
    basewidth = 256
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(INPUT_DIR + "1.jpg", "JPEG")
    run_cmd("python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input "+ INPUT_DIR + " --output " + OUTPUT_DIR + " --netscale 4 --outscale 3.5")
    return os.path.join(OUTPUT_DIR, "1_out.jpg")
  
title = "Anime2Sketch"
description = "demo for Anime2Sketch. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2104.05703'>Adversarial Open Domain Adaption for Sketch-to-Photo Synthesis</a> | <a href='https://github.com/Mukosame/Anime2Sketch'>Github Repo</a></p>"
gr.Interface(
    inference, 
    [gr.inputs.Image(type="pil", label="Input")], 
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
).launch(debug=True)

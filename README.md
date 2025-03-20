# Diffusers Client

Windows11 + WSL2

---

## python

```
$ python --version
3.10.6
```

---

## installation

```
$ python -m venv .venv
```

```
$ source ./.venv/bin/activate
```

```
$ (.venv) pip install diffusers transformers peft compel xformers fastapi uvicorn python-dotenv
```

---

### checking

```
$ (.venv) python
$ import torch
$ print(torch.cuda.is_available())
True
```

---

### version check

```
$ (.venv) pip freeze
accelerate==1.5.2
annotated-types==0.7.0
anyio==4.9.0
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
compel==2.0.3
diffusers==0.32.2
exceptiongroup==1.2.2
fastapi==0.115.11
filelock==3.18.0
fsspec==2025.3.0
h11==0.14.0
huggingface-hub==0.29.3
idna==3.10
importlib_metadata==8.6.1
Jinja2==3.1.6
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.4
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-cusparselt-cu12==0.6.2
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
packaging==24.2
peft==0.14.0
pillow==11.1.0
psutil==7.0.0
pydantic==2.10.6
pydantic_core==2.27.2
pyparsing==3.2.1
python-dotenv==1.0.1
PyYAML==6.0.2
regex==2024.11.6
requests==2.32.3
safetensors==0.5.3
sniffio==1.3.1
starlette==0.46.1
sympy==1.13.1
tokenizers==0.21.1
torch==2.6.0
tqdm==4.67.1
transformers==4.49.0
triton==3.2.0
typing_extensions==4.12.2
urllib3==2.3.0
uvicorn==0.34.0
xformers==0.0.29.post3
zipp==3.21.0
```

---

## Environment Variable

```
$ touch .env
```

```.env
MODEL_FILE_PATH=<full path of the model file>
```

---

## start API

```
$ (.venv) uvicorn main:app --reload
```

---

### API call sample

curl http://127.0.0.1:8000/image \
-X POST \
-H 'Content-Type:application/json' \
-d '{"prompt": "1girl, (masterpiece)1.1, amazing quality, volumetric lighting", "ng_prompt":"low quality, bad quality", "width": 1024, "height": 1024, "seed": 9999, "file_name":"girl_0001.png"}'

---

### kill process

Ctrl + Z 

```
$ sudo aux | grep uvicorn
```

```
$ sudo kill -9 <process_id>
```

---

## Reference

* [猫耳とdiffusersで始めるStable Diffusion入門](https://qiita.com/phyblas/items/00f750b8277f66fb9b13)
* [HuggingFace ブログ : SDXL のための単純な最適化の探求](https://torch.classcat.com/2023/11/03/huggingface-blog-simple-sdxl-optimizations/)
* [Diffusersで画像生成する](https://qiita.com/jcomeme/items/d9953b4bc2212c199d16)

---

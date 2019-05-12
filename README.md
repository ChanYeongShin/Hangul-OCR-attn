# Hangul-OCR
한글 OCR (attention-based)

attention을 이용한 CNN+LSTM encoder / LSTM attention decoder repo입니다. ([RARE paper](https://arxiv.org/abs/1603.03915))

실험 과정 및 결과들에 대한 일지는 하단의 링크를 참고해주세요.
([링크](https://mathpresso.atlassian.net/wiki/spaces/RES/pages/297500798/Hangul+OCR))

## Prerequsite
### How to install

`$pip install -r requirements.txt`

## run with GPU

`$CUDA_VISIBLE_DEVICES=0 python train.py with config_template.json`

GPU 번호는 nvidia-smi로 먼저 사용하고 있는 GPU를 확인하시고 유동적으로 바꾸어주세요.

## Data

### which type csv have
csv 는 filename, width, height, input_text, xmin, ymin, xmax, ymax의 형태로 저장되어있습니다. 

filename의 file을 읽어와 ltrb(left, right, top, bottom) 정보를 이용해 box cropping 과정을 거치고 난 데이터를 이용합니다.

### what kind of input 
input은 이 csv를 line by line으로 읽어서 집어넣기 때문에 data/csv만 있으면 됩니다.



**csv 및 label map json 정제에 관한 jupyter notebook은 ([CRNN repo](https://github.com/mathpresso/Hangul-OCR-crnn)) repo notebook을 참고해주세요.


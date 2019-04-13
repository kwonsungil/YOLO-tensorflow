# YOLO-tensorflow
You Only Look Once 모델 v1~v3 구현 by tensorflow


<h2>YOLO v1</h2>
 
<h3>Pascal VOC dataset</h3>

<h3>전처리</h3>

utils/preprocess_pascal_voc.py 파일 실행

cofig 파일 안에 PASCAL_DIR, YEAR 을 이용해서 xml이 들어 있는 directory 지정

결과 : data/voc_train_(year).txt 생성

 
<h3>학습</h3>
 
pretrrain 모델 사용 =>  = "./data_set/YOLO_small.ckpt"

YOLO_small.ckpt download : https://drive.google.com/file/d/1NqIkvMGnuTJS_WoAP8Yr92GNueDxbPRW/view?usp=sharing
 
yolo_v1_train.py 실행

(voc_train_year.txt 파일 읽어서 numpy 배열로 바꾼 후 저장 후 train)
 

 
 
<h2>YOLO v2</h2>

 : 예정


<h2>YOLO v3</h2>

: 예정


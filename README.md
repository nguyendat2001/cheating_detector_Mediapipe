
## Installation

install environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# pip install required packages
pip install requirement.txt
```

</details>

## make data 


``` shell
python make_data.py --data ./data/train --type train
```

## Training

Data preparation

``` shell
python train.py --train ./data/landmarks/train --test ./data/landmarks/test --output ./figures --epochs 500 --batch 64 --patience 20
```

## inference head pose detection 
detect several pose of head include: looking left, right, up, down, forward, talking with somebody 

``` shell
python face_mesh_test.py --data ./video.mp4 
python face_mesh_test.py
```

## inference test model body pose
detect cheating or non_cheating behaved of examinee

``` shell
# python test_model_pose.py --data test.mp4 --model ./figures/model.h5
# python test_model_pose.py --model ./figures/model.h5
```


<div align="center">
    <a href="./">
        <img src="" width="59%"/>
    </a>
</div>

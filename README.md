# Temporal Relation Networks

We release the code of the [Temporal Relation Networks](http://relation.csail.mit.edu/), built on top of the [TSN-pytorch codebase](https://github.com/yjxiong/temporal-segment-networks).

**Note**: always use `git clone --recursive https://github.com/mitrydoug/TRN-pytorch` to clone this project
Otherwise you will not be able to use the inception series CNN architecture.

![framework](http://relation.csail.mit.edu/framework_trn.png)

### Data preparation
Download the [something-something dataset](https://www.twentybn.com/datasets/something-something) or [jester dataset](https://www.twentybn.com/datasets/something-something) or [charades dataset](http://allenai.org/plato/charades/). Decompress them into some folder. Use [process_dataset.py](process_dataset.py) to generate the index files for train, val, and test split. Finally properly set up the train, validatin, and category meta files in [datasets_video.py](datasets_video.py).

### Code

Core code to implement the Temporal Relation Network module is [TRNmodule](TRNmodule.py). It is plug-and-play on top of the TSN.

### Training and Testing

* The command to train single scale TRN

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
                     --arch BNInception --num_segments 3 \
                     --consensus_type TRN --batch-size 64
```

* The command to train multi-scale TRN
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type TRNmultiscale --batch-size 64
```

* The command to test the single scale TRN

```bash
python test_models.py something RGB model/TRN_something_RGB_BNInception_TRN_segment3_best.pth.tar \
   --arch BNInception --crop_fusion_type TRN --test_segments 3
```

* The command to test the multi-scale TRN

```bash
python test_models.py something RGB model/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
   --arch BNInception --crop_fusion_type TRNmultiscale --test_segments 8
```

### Pretrained models and demo code

* Download pretrained models on [Something-Something](https://www.twentybn.com/datasets/something-something), [Jester](https://www.twentybn.com/datasets/jester), and [Moments in Time](http://moments.csail.mit.edu/)

```bash
cd pretrain
./download_models.sh
```

* Download sample video and extracted frames. There will be mp4 video file and a folder containing the RGB frames for that video.

```bash
cd sample_data
./download_sample_data.sh
```

The sample video is the following 
![result](http://relation.csail.mit.edu/data/bolei_juggling.gif): Bolei is juggling:) 

* Test pretrained model trained on Something-Something

```bash
python test_video.py --arch BNInception --dataset something \
    --weight pretrain/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
    --frame_folder sample_data/bolei_juggling

RESULT ON sample_data/bolei_juggling
0.244 -> Throwing something in the air and catching it
0.186 -> Throwing something in the air and letting it fall
0.094 -> Showing a photo of something to the camera
0.063 -> Hitting something with something
0.040 -> Holding something in front of something

```


* Test pretrained model trained on [Moments in Time](http://moments.csail.mit.edu/)

```bash
python test_video.py --arch InceptionV3 --dataset moments \
    --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar \
    --frame_folder sample_data/bolei_juggling

RESULT ON sample_data/bolei_juggling

0.982 -> juggling
0.003 -> flipping
0.003 -> spinning
0.003 -> smoking
0.002 -> whistling
```

* Test pretrained model on mp4 video file

```bash
python test_video.py --arch InceptionV3 --dataset moments \
    --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar \
    --video_file sample_data/bolei_juggling.mp4 --rendered_output sample_data/predicted_video.mp4 
```

The command above uses `ffmpeg` to extract frames from the supplied video `--video_file` and optionally generates a new video `--rendered_output` from the frames used to make the prediction with the predicted category in the top-left corner.


### TODO

* TODO: Web-cam demo script
* TODO: Visualization script
* TODO: class-aware data augmentation

### Reference:
B. Zhou, A. Andonian, and A. Torralba. Temporal Relational Reasoning in Videos. arXiv:1711.08496, 2017. [PDF](https://arxiv.org/pdf/1711.08496.pdf)
```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Torralba, Antonio},
    journal={arXiv:1711.08496},
    year={2017}
}
```

### Acknowledgement
Our temporal relation network is plug-and-play on top of the [TSN-Pytorch](https://github.com/yjxiong/temporal-segment-networks), but it could be extended to other network architectures easily. We thank Yuanjun Xiong for releasing TSN-Pytorch codebase. Something-something dataset and Jester dataset are from [TwentyBN](https://www.twentybn.com/), we really appreciate their effort to build such nice video datasets. Please refer to [their dataset website](https://www.twentybn.com/datasets/something-something) for the proper usage of the data.

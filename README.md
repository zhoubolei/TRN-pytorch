# Temporal Relation Networks 

This is the codebase of the temporal relation networks, built on top of the [TSN-pytorch codebase](https://github.com/yjxiong/temporal-segment-networks).

**Note**: always use `git clone --recursive https://github.com/metalbubble/TRN-TSN` to clone this project. 
Otherwise you will not be able to use the inception series CNN archs. 


The command to train single scale TRN

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
                     --arch BNInception --num_segments 3 \
                     --consensus_type TRN --batch-size 64
```

The command to train multi-scale TRN
```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py something RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type TRNmultiscale --batch-size 64
```




--------------------
# Below is the README from TSN-Pytorch

## Training

To train a new model, use the `main.py` script.

The command to reproduce the original TSN experiments of RGB modality on UCF101 can be 

```bash
python main.py ucf101 RGB <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

For flow models:

```bash
python main.py ucf101 Flow <ucf101_flow_train_list> <ucf101_flow_val_list> \
   --arch BNInception --num_segments 3 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 128 -j 8 --dropout 0.7 \
   --snapshot_pref ucf101_bninception_ --flow_pref flow_  
```

For RGB-diff models:

```bash
python main.py ucf101 RGBDiff <ucf101_rgb_train_list> <ucf101_rgb_val_list> \
   --arch BNInception --num_segments 7 \
   --gd 40 --lr 0.001 --lr_steps 80 160 --epochs 180 \
   -b 128 -j 8 --dropout 0.8 \
   --snapshot_pref ucf101_bninception_ 
```

## Testing

After training, there will checkpoints saved by pytorch, for example `ucf101_bninception_rgb_checkpoint.pth`.

Use the following command to test its performance in the standard TSN testing protocol:

```bash
python test_models.py ucf101 RGB <ucf101_rgb_val_list> ucf101_bninception_rgb_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name>

```

Or for flow models:
 
```bash
python test_models.py ucf101 Flow <ucf101_rgb_val_list> ucf101_bninception_flow_checkpoint.pth \
   --arch BNInception --save_scores <score_file_name> --flow_pref flow_

```

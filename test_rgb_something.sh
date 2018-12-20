#!/usr/bin/env bash
python test_models.py \
    something \
    RGB \
    model/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
   --arch BNInception \
   --crop-fusion-type TRNmultiscale \
   --test-segments 8

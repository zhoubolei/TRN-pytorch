#!/bin/sh
# Download the pre-trained TRN models
echo 'Downloading TRNmultiscale on Something-Something'
wget http://relation.csail.mit.edu/models/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar

echo 'Downloading TRNmultiscale on Jester'
wget http://relation.csail.mit.edu/models/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar

echo 'Download TRNmultiscale on Moments in Time'
wget http://relation.csail.mit.edu/models/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar

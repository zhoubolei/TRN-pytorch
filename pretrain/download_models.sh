#!/bin/sh
# Download the pre-trained TRN models
echo 'Downloading TRNmultiscale on Something-Something'
wget http://relation.csail.mit.edu/models/TRN_something_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar
wget http://relation.csail.mit.edu/models/TRN_something_RGB_BNInception_TRN_segment3_best.pth.tar
wget http://relation.csail.mit.edu/models/something_categories.txt

echo 'Downloading TRNmultiscale on Jester'
wget http://relation.csail.mit.edu/models/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar
wget http://relation.csail.mit.edu/models/jester_categories.txt

echo 'Downloading TRNmultiscale on Moments in Time'
wget http://relation.csail.mit.edu/models/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar
wget http://relation.csail.mit.edu/models/moments_categories.txt



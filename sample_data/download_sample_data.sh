echo 'Downloading sample test video and extracted frames'
wget http://relation.csail.mit.edu/data/juggling.mp4
wget -r -nH --cut-dirs=1 --no-parent --reject="index.html*"  http://relation.csail.mit.edu/data/juggling_frames/

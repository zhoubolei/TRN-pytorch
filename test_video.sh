# Make prediction from mp4 video file.
python test_video.py --video_file sample_data/juggling.mp4 --rendered_output sample_data/predicted_video.mp4

# Make prediction using list of extracted video frames.
python test_video.py --frame_list sample_data/juggling_frame_list.txt

import csv
import os
root_jpg = 'Charades_v1_rgb'

with open('Charades_v1_classes.txt') as f:
    lines = f.readlines()

output_categories = []
dict_class2idx = {}
for i, line in enumerate(lines):
    line = line.rstrip()
    items = line.split()
    class_id = items[0]
    label = ' '.join(items[1:])
    output_categories.append(label)
    dict_class2idx[class_id] = i

with open('categories.txt','w') as f:
    f.write('\n'.join(output_categories))

def process_split(filename):
    fps = 24
    output_frameno = []
    output_segments = []

    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['id']
            actions = row['actions'].split(';')
            dir_files = os.listdir(os.path.join(root_jpg, video_id))
            num_frames = len(dir_files)
            output_frameno.append('%s %d'%(video_id, num_frames))

            print('%s %d images' % (video_id, len(dir_files)))
            for action in actions:
                items = action.split()
                if len(items)>0:
                    id_action = dict_class2idx[items[0]]
                    start_frame = max(int(min(float(items[1])*fps, num_frames)),1)
                    end_frame = int(min(float(items[2])*fps, num_frames))
                    if end_frame-start_frame > 5:
                        output_segments.append('%s %d %d %d' % (video_id, start_frame, end_frame, id_action))
    return output_frameno, output_segments
splits = ['test','train']
for split in splits:
    filename = 'Charades_v1_%s.csv' % split
    filename_output = '%s_segments.txt' % split
    filename_frameno = '%s_frameno.txt' % split
    output_video_frameno, output_segments = process_split(filename)

    with open(filename_output,'w') as f:
        f.write('\n'.join(output_segments))
    with open(filename_frameno,'w') as f:
        f.write('\n'.join(output_video_frameno))



import os
import sys

root = None
backup = None

def process_split(split):
    split_file = f'{root}{split}Set.csv'
    frames_file = f'{root}{split}_frames.txt'

    with open(f'{root}moments_categories.txt', 'r') as f:
        labels = dict(tuple(line.strip().split(',')) for line in f)

    i = 0
    
    with open(split_file, 'r') as f:
        with open(frames_file, 'w') as g:
            for line in f:
                fn, label, _, _ = line.split(',')
                assert (fn[-4:] == '.mp4')
                if not os.path.isdir(f'{root}{split}/{fn[:-4]}'):
                    if backup is not None and \
                            os.path.isdir(f'{backup}{split}/{fn[:-4]}'):
                        os.system(f'cp -r {backup}{split}/{fn[:-4]} {root}{split}/{fn[:-4]}')
                    else:
                        os.system(f'mkdir {root}{split}/{fn[:-4]} && \
                                    ffmpeg -loglevel panic -i {root}{split}/{fn} \
                                        -f image2 {root}{split}/{fn[:-4]}/%5d.jpg')
                num_frames = len(os.listdir(f'{root}{split}/{fn[:-4]}'))
                g.write(f'{split}/{fn[:-4]} {num_frames} {labels[label]}\n')
                i += 1
                print('%5d: %s' % (i, fn))

def main():
    global root, backup
    root = sys.argv[1] # /home/shared/Moments_in_Time_Mini_Stream/
    if root[-1] != '/':
        root += '/'
    if len(sys.argv) > 2:
        backup = sys.argv[2]
        if backup[-1] != '/':
            backup += '/'
    process_split('training')
    process_split('validation')

if __name__ == '__main__':
    main()

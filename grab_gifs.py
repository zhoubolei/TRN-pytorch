import os
import pandas as pd
import moviepy.editor as mov_editor
import logging
from time import sleep
from multiprocessing import Pool
from PIL import Image
from itertools import islice
import pickle
from math import ceil

logging.basicConfig(
    format='%(asctime)s:%(levelname)s: %(message)s',
    level=logging.INFO
)

OUTPUT_PATH = '/home/ec2-user/gifs'

# gifs2cat = pd.read_csv('/home/ec2-user/gifs2cat.csv')
# gifs2cat = gifs2cat.groupby('gif_id')['category'].apply(list)
# gifs2cat = pd.read_csv('/home/ec2-user/reactions.csv')
payload = pickle.load(open('/home/ec2-user/siamese_dataset.pkl', 'rb'))
gifs = []
for x in payload:
    gifs.extend(list(x))


def evenly_spaced_sampling(array, n):
    """Choose `n` evenly spaced elements from `array` sequence"""
    length = len(array)

    if n == 0 or length == 0:
        return []
    elif n == length:
        return array
    elif n < length:
        return [array[ceil(i * length / n)] for i in range(n)]
    elif n > length:
        result = []
        for _ in range(ceil(n / length)):
            result.extend(array)
        return result[:n]

    
def get_gif_mov_url(id, ext='mp4'):
    return f'https://media.giphy.com/media/{id}/giphy.{ext}'


def get_gif(id):
    gif = None
    for i in range(5):
        try:
            gif = mov_editor.VideoFileClip(get_gif_mov_url(id))
            break
        except Exception as ex:
            sleep(1)
            logging.info(f'Error: {type(ex)}:{ex}. {i} times. With extension mp4.')
    else:
        for i in range(5):
            try:
                gif = mov_editor.VideoFileClip(get_gif_mov_url(id, ext='gif'))
                break
            except Exception as ex:
                sleep(1)
                logging.info(f'Error: {type(ex)}:{ex}. {i} times. With extension gif.')
    return gif


def save_gif(id):
    directory = os.path.join(OUTPUT_PATH, id)
    if os.path.isdir(directory):
        return None
    gif = get_gif(id)
    if gif is not None:
        os.mkdir(directory)
        frames = list(gif.iter_frames())
        for i, x in enumerate(evenly_spaced_sampling(frames, 50)):
            Image.fromarray(x).save(os.path.join(directory, f'{i}.jpg'))
        del gif

        
print('Total gifs:', len(gifs))
with Pool(processes=16) as executor:
    executor.map(save_gif, gifs)
#     executor.map(save_gif, gifs2cat.index)

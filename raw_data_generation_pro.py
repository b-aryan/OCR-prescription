from PIL import Image, ImageDraw, ImageFont
import textwrap
import nltk
import os
import random
import warnings
from image_processing import Image_Processing

warnings.filterwarnings(action='ignore')

from nltk.corpus import reuters
corpus = reuters.sents()

file_path = os.path.dirname(os.path.abspath(__file__))

num_text = 500
len_text = random.randint(10, 100)
text_samples =  [" ".join(sent) for sent in corpus[:num_text]]

'''common used four fonts and their transformers'''

font_set = ['arial.ttf', 'arialbd.ttf', 'arialbi.ttf',
            'ariali.ttf', 'ARIALN.TTF', 'ARIALNB.TTF',
            'ARIALNBI.TTF', 'ARIALNI.TTF', 'ariblk.ttf',
            'times.ttf', 'timesbd.ttf', 'timesbi.ttf',
            'timesi.ttf', 'calibri.ttf', 'calibrib.ttf',
            'calibrii.ttf', 'calibril.ttf', 'calibrili.ttf',
            'calibriz.ttf', 'CALIFB.TTF', 'CALIFI.TTF',
            'CALIFR.TTF', 'CALIST.TTF', 'CALISTB.TTF', 'CALISTBI.TTF',
            'CALISTI.TTF', 'verdana.ttf', 'verdanab.ttf', 'verdanai.ttf',
            'verdanaz.ttf', 'GARA.TTF', 'GARABD.TTF',
            'GARAIT.TTF', 'cambria.ttc', 'cambriab.ttf',
            'cambriai.ttf', 'cambriaz.ttf', 'CENSCBK.TTF',
            'CENTAUR.TTF', 'CENTURY.TTF', 'tahoma.ttf', 'tahomabd.ttf']

color_set = [(255, 255, 255), # White
            (173, 216, 230), # Light Blue
            (255, 182, 193), # Light Pink
            (255, 255, 224), # Light Yellow
            (144, 238, 144), # Light Green
            (255, 222, 173), # Light Orange
            (230, 230, 250), # Light Purple
            (211, 211, 211), # Light Gray
            (222, 184, 135)] # Light Brown


for idx, ele in enumerate(text_samples):
    len_text = random.randint(10, 100)
    '''create ImageDraw instance'''
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    '''font format'''
    font_size = random.randint(14,24)
    random_font = random.choice(font_set)

    selected_font = ImageFont.truetype(random_font, font_size )

    # padding = random.randint(1,20)
    padding = 10
    wrapped_text = textwrap.fill(ele[0:len_text], width = 120)
    wrapped_lines = wrapped_text.split('\n') # to generate raw data set with multiline , if needed

    text_width, text_height = draw.textsize(wrapped_text, font = selected_font)
    canvas_width = text_width + 2 * padding
    canvas_height = text_height + 2 * padding

    random_color = random.choice(color_set)
    canvas = Image.new("RGB", (canvas_width, canvas_height), random_color)

    draw = ImageDraw.Draw(canvas)
    x_offset =  padding
    y_offset =  (canvas_height - text_height) // 2

    for line in wrapped_lines:
        draw.text((x_offset, y_offset), line, fill=(0, 0, 0), font=selected_font)
        y_offset += selected_font.getsize(line)[1]

    label_path = os.path.join(file_path, 'data_set','labels',f'label_{idx + 1}.txt')

    image_path = os.path.join(file_path, 'data_set', 'images')
    image_processed = Image_Processing(image_path)
    grayed_image_list = image_processed.gray_scale_conver(canvas ,is_save = False )[1]
    image_processed.split_image(grayed_image_list, num = f'{idx + 1}')

    with open(label_path, "w", encoding="utf-8") as file:
        file.write(ele[0:len_text])

    print(f'generation: {((idx+1)/len(text_samples)*100):.2f}%', end = '\n')
pass

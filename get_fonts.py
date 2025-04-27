import os
import fnmatch

def find_files(folder, keyword):
    result = []
    for root, dirs, files in os.walk(folder):
        for file in fnmatch.filter(files, '*' + keyword + '*'):
            result.append(os.path.join(root, file))
    return result

folder_path = r'C:\Windows\Fonts'  # 替换为实际的文件夹路径
main_font_list = [
    'Ari', # Arial
    'Hel', # Helvetica
    'Tim', # Times New Roman
    'Cali', # Calibri
    'Ver', # Verdana
    'Gar', # Garamond
    'Cam', # Cambria
    'Cen', # Century Gothic
    'Tah' # Tahoma
]

font_list = []
for font in main_font_list:
    files_with_keyword = find_files(folder_path,font)
    for file in files_with_keyword:
        font_list.append(os.path.basename(file))

print(font_list)
print(len(font_list))

['arial.ttf', 'arialbd.ttf', 'arialbi.ttf',
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

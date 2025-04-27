import pickle
import glob
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
alphabet_path = os.path.join(abs_path,  'alphabet.pkl')
if os.path.isfile(alphabet_path):
    # print('test for only load the .pkl file')

    alphabet_list = pickle.load(open(alphabet_path, 'rb'))
    alphabet = [ord(char) for char in alphabet_list]
    alphabet_v2 = alphabet

else:
    # print('test for generating .pkl file')

    '''get alphabets from labels data set'''
    alphabet_set = set()
    # labels_path = r'E:\00.HKU\03.Courses\04.Capstone\00.Coding\01.OCR\labels'
    # git_file_path =  r'E:\00.HKU\03.Courses\04.Capstone\00.Coding\01.OCR\utils\alphabet_git.pkl'
    labels_path = os.path.join(os.path.dirname(abs_path), 'data_set',  'labels')
    git_file_path = os.path.join(abs_path, 'alphabet_git.pkl')

    txt_files = glob.glob(os.path.join(labels_path, '*.txt'))
    for idx, file in enumerate(txt_files):
        with open(file, encoding='utf-8') as f:
            content = f.readlines()
            f.close()
        for line in content:
            for char in line:
                alphabet_set.add(char)

    print(f'gain {len(alphabet_set)} alphabets from labels')

    '''get alphabets from git'''
    with open(git_file_path, 'rb') as file:
        git_data = pickle.load(file)

    '''union these two'''
    alphabet_set = alphabet_set.union(git_data)

    print(f'gain {len(alphabet_set)} alphabets totally')

    '''save the .pkl file'''
    alphabet_list = sorted(list(alphabet_set))
    pickle.dump(alphabet_list,open(alphabet_path,'wb'))

    alphabet_list = pickle.load(open(alphabet_path,'rb'))
    alphabet = [ord(ch) for ch in alphabet_list]
    alphabet_v2 = alphabet
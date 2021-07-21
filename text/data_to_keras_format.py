import os

out_path = './data_keras_format/'
data_path = './data/'

"""
data_keras_format/
    decoy/
        train/
            class_a/
                1.txt
                2.txt
            class_b/
                1.txt
                2.txt
        dev/
            class_a/
                1.txt
                2.txt
            class_b/
                1.txt
                2.txt
"""


class_names = ['negative', 'positive']
dirs = os.listdir(data_path)
for dir_ in dirs:
    files = os.listdir(data_path+'/'+dir_)
    for file in files:
        split, noise_type, _ = file.split('.')[0].split('_')
        f = open(data_path+dir_+'/'+file, 'r')
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            text, label  = line[:-2], line[-1]
            label = int(label)
            save_path = out_path+noise_type+'/'+split+'/'+class_names[label]+'/'
            os.makedirs(save_path, exist_ok=True)
            fw = open(save_path+str(idx)+'.txt', 'w')
            fw.write(text)
            fw.close()

            
        
        
        
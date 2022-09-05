import shutil
import os

path = '/media/jingcheng/Disk G/KittiDataset'
folders = os.listdir(path)

for folder in folders:
    #print(folder)
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    for file in files:
        print(file)
        filepath = os.path.join(folder_path, file)
        sfiles = os.listdir(filepath)
        for sfile in sfiles:
            try:
                ssfiles = os.listdir(os.path.join(filepath, sfile))
                for ssfile in ssfiles:
                    #print(os.path.join(filepath, sfile))
                    shutil.move(os.path.join(filepath, sfile, ssfile), os.path.join(path, file, sfile))
            except:
                print("wRONG ", sfile)
    #shutil.rmtree(folder_path)


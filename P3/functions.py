import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path
import numpy as np
from chainCodes import contour, show_img

def clean_console():
  os.system('cls' if os.name == 'nt' else 'clear')

def open_img(imgData:list):
    if imgData[1].split('.')[1] == 'obj':
        open_3d_img(imgData)
        return
    img = mpimg.imread(imgData[1])
    plt.imshow(img, cmap='gray')
    plt.title(imgData[0])
    plt.axis('off')
    plt.show()

def get_binary_data(data):
    binary_data = []
    voxels_data = data[1].split('\n')[0].split(' ')[1:]
    x, y, z = int(voxels_data[0]), int(voxels_data[1]), int(voxels_data[2])
    data = data[5:]
    data = [d.split('\n')[0] for d in data]

    for i in range(0, len(data), x):
        binary_data.append(data[i:i+x])

    new_binary_data = np.ones((x, y, z), dtype=np.int8)
    for i, d in enumerate(binary_data):
        for j, b in enumerate(d):
            arr = np.array([int(c) for c in b])
            new_binary_data[i][j] = arr

    return new_binary_data


def binary_to_scr(binary_data, name:str):
    scr_file = Path(f'{name}.scr')
    if os.path.isfile(scr_file):
        os.remove(scr_file)

    with open(scr_file, 'w') as f:
        coords = np.transpose(np.nonzero(binary_data))
        count = 0
        for z, y, x in coords:
            print(f'z: {z}, y: {y}, x: {x}')
            print(f'{count}/{coords.size}')
            f.write(f"_box\nc\n{x+.5},{y+.5},{z+.5}\nc\n1\n")
            count += 1
            clean_console()
        f.write("\n")

                        
    os.system(f'acad.exe {scr_file}')

def open_3d_img(imgData:list, ret:bool = False):
    file = Path(imgData[1])
    filedirectory = str(file.parent.absolute())
    try:
        os.chdir(filedirectory)
    except Exception as e:
        print(e)
    if not os.path.isfile(f'{file.name.split(".")[0]}.binvox'):
        os.system(f'binvox.exe -d 100 {file.name}')
    if not ret:
        os.system(f'viewvox.exe {file.name.split(".")[0]}.binvox')

    os.system(f'convertidor.exe {file.name.split(".")[0]}.binvox')

    binary_file = Path(f'voxels.txt')

    data = []

    with open(binary_file, 'r') as f:
        data = f.readlines()

    os.remove(binary_file)
    
    binary_data = get_binary_data(data)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if ret:
        return binary_data

def save_info():
    print("Para guardar la informacion")
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, rgba2rgb

def neighborhoodCoordinates(row, col, size, img_shape):
    coordinates = []
    ret = []
    if size == 4:
        coordinates = [(row, col+1, 0), (row+1, col, 1), (row, col-1, 2), (row-1, col, 3)]
    elif size == 8:
        coordinates = [(row, col+1, 0), (row+1, col+1, 1), (row+1, col, 2), (row+1, col-1, 3) \
                       ,(row, col-1, 4), (row-1, col-1, 5), (row-1, col, 6), (row-1, col+1, 7)]
    for coord in coordinates:
        if coord[0] < 0 or coord[1] < 0 or coord[0] >= img_shape[0] or coord[1] >= img_shape[1]:
            continue
        else:
            ret.append(coord)
    return ret

def binarizar(img, w_opt, b_opt):
    return np.where(img >= 155, w_opt, b_opt)
  
def dilatation(img, k_size):
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for row, col in np.ndindex(img.shape):
        neighboors = neighborhoodCoordinates(row, col, k_size, img.shape)
        for coord in neighboors:
            if img[coord[0], coord[1]] == 1:
                newimg[row, col] = 1
                break
    return newimg

def erosion(img, k_size):
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for row, col in np.ndindex(img.shape):
        neighbors = neighborhoodCoordinates(row, col, k_size, img.shape)
        sum = 0
        for coord in neighbors:
            sum += img[coord[0], coord[1]]
        if sum == k_size:
            newimg[row, col] = 1
    return newimg

def freeman4ChainCode(img):
    aux = []
    chain = []
    y = x = None
    shape = (img.shape[0], img.shape[1])
    for row, col in np.ndindex(shape):
        if img[row, col][0] == 1:
            y = row
            x = col
            break
    img[y, x][0] = 0

    if img[y][x - 1][1] != 1:
        aux.append(3)
        if img[y - 1][x - 1][1] != 1:
            if img[y - 1][x][1] != 1:
                chain.append(0)
                if img[y - 1][x + 1][1] != 1:
                    if img[y][x + 1][1] != 1:
                        chain.append(1)
    while True:
        if img[y][x+1][0] == 1:
            img[y][x+1][0] = 0
            x += 1
            if img[y - 1][x][1] != 1:
                chain.append(0)
                if img[y - 1][x + 1][1] != 1:
                    if img[y][x + 1][1] != 1:
                        chain.append(1)
                        if img[y + 1][x + 1][1] != 1:
                            if img[y + 1][x][1] != 1:
                                chain.append(2)

        elif img[y+1][x][0] == 1:
            img[y+1][x][0] = 0
            y += 1
            if img[y][x + 1][1] != 1:
                chain.append(1)
                if img[y + 1][x + 1][1] != 1:
                    if img[y + 1][x][1] != 1:
                        chain.append(2)
                        if img[y + 1][x - 1][1] != 1:
                            if img[y][x-1][1] != 1:
                                chain.append(3)
        
        elif img[y][x-1][0] == 1:
            img[y][x-1][0] = 0
            x -= 1
            if img[y + 1][x][1] != 1:
                chain.append(2)
                if img[y + 1][x - 1][1] != 1:
                    if img[y][x-1][1] != 1:
                        chain.append(3)
                        if img[y - 1][x - 1][1] != 1:
                            if img[y-1][x][1] != 1:
                                chain.append(0)
        
        elif img[y-1][x][0] == 1:
            img[y-1][x][0] = 0
            y -= 1
            if img[y][x - 1][1] != 1:
                chain.append(3)
                if img[y - 1][x - 1][1] != 1:
                    if img[y - 1][x][1] != 1:
                        chain.append(0)
                        if img[y - 1][x + 1][1] != 1:
                            if img[y][x+1][1] != 1:
                                chain.append(1)
        
        elif img[y-1][x+1][0] == 1:
            img[y-1][x+1][0] = 0
            y -= 1
            x += 1
            if img[y][x - 1][1] != 1:
                chain.append(3)
                if img[y - 1][x - 1][1] != 1:
                    if img[y - 1][x][1] != 1:
                        chain.append(0)
                        if img[y - 1][x + 1][1] != 1:
                            if img[y][x + 1][1] != 1:
                                chain.append(1)
                                if img[y + 1][x + 1][1] != 1:
                                    if img[y + 1][x][1] != 1:
                                        chain.append(2)

        elif img[y+1][x+1][0] == 1:
            img[y+1][x+1][0] = 0
            y += 1
            x += 1
            if img[y - 1][x][1] != 1:
                chain.append(0)
                if img[y - 1][x + 1][1] != 1:
                    if img[y][x + 1][1] != 1:
                        chain.append(1)
                        if img[y + 1][x + 1][1] != 1:
                            if img[y + 1][x][1] != 1:
                                chain.append(2)
                                if img[y + 1][x - 1][1] != 1:
                                    if img[y][x-1][1] != 1:
                                        chain.append(3)

        elif img[y+1][x-1][0] == 1:
            img[y+1][x-1][0] = 0
            y += 1
            x -= 1
            if img[y][x + 1][1] != 1:
                chain.append(1)
                if img[y + 1][x + 1][1] != 1:
                    if img[y + 1][x][1] != 1:
                        chain.append(2)
                        if img[y + 1][x - 1][1] != 1:
                            if img[y][x-1][1] != 1:
                                chain.append(3)
                                if img[y - 1][x - 1][1] != 1:
                                    if img[y-1][x][1] != 1:
                                        chain.append(0)

        elif img[y-1][x-1][0] == 1:
            img[y-1][x-1][0] = 0
            y -= 1
            x -= 1
            if img[y + 1][x][1] != 1:
                chain.append(2)
                if img[y + 1][x - 1][1] != 1:
                    if img[y][x - 1][1] != 1:
                        chain.append(3)
                        if img[y - 1][x - 1][1] != 1:
                            if img[y - 1][x][1] != 1:
                                chain.append(0)
                                if img[y - 1][x + 1][1] != 1:
                                    if img[y][x+1][1] != 1:
                                        chain.append(1)
        else:
            break
    while len(aux)>0:
        chain.append(aux.pop(0))
    return chain

def isBorderPixel(row, col, img, search_size):
    neighboors = neighborhoodCoordinates(row, col, search_size, img.shape)
    for coord in neighboors:
        if img[coord[0], coord[1]] == 0:
            return True
    return False

def freemanChainCodeAlgorithm(row, col, img, size, p_pixel=None):
    neighboors = neighborhoodCoordinates(row, col, size, img.shape)
    for coord in neighboors:
        if img[coord[0], coord[1]] == 1:
            if size == 8:
                if isBorderPixel(coord[0], coord[1], img, 4):
                    if (coord[0], coord[1]) not in p_pixel:
                        return coord[2], (coord[0], coord[1])
def freeman8ChainCode(img):
    coords = []
    chain = []
    p_pixels = []
    n_pixel = f_pixel = None
    for row, col in np.ndindex(img.shape):
        if img[row, col] == 1:
            if isBorderPixel(row, col, img, 4):
                f_pixel = (row, col)
                ret = freemanChainCodeAlgorithm(row, col, img, 8, p_pixels)
                chain.append(ret[0])
                coords.append(ret[1])
                n_pixel = ret[1]
                break
    while True:
        try:
            ret = freemanChainCodeAlgorithm(n_pixel[0], n_pixel[1], img, 8, p_pixels)
            coords.append(ret[1])
            chain.append(ret[0])
            p_pixels.append(n_pixel)
            n_pixel = ret[1]
            if n_pixel == f_pixel:
                break
        except Exception:
            break
    return chain, coords

def VCC(f4_chain):
    vcc = []
    convertion = [[ 0 , 1 ,"*", 2 ],
                  [ 2 , 0 , 1 ,"*"],
                  ["*", 2 , 0 , 1 ],
                  [ 1 ,"*", 2 , 0 ]]
    
    for i in range(len(f4_chain) - 1):
        vcc.append(convertion[f4_chain[i]][f4_chain[i+1]])
    return vcc

def cAF8(f8_chain):
    af8 = []
    convertion = [[0, 1, 2, 3, 4, 5, 6, 7],
                  [7, 0, 1, 2, 3, 4, 5, 6],
                  [6, 7, 0, 1, 2, 3, 4, 5],
                  [5, 6, 7, 0, 1, 2, 3, 4],
                  [4, 5, 6, 7, 0, 1, 2, 3],
                  [3, 4, 5, 6, 7, 0, 1, 2],
                  [2, 3, 4, 5, 6, 7, 0, 1],
                  [1, 2, 3, 4, 5, 6, 7, 0]]

    for i in range(len(f8_chain) - 1):
        af8.append(convertion[f8_chain[i]][f8_chain[i+1]])
    af8.append(2)
    return af8

def c3OT(f4_chain):
    c3ot = []
    aux = 1
    convertion = [[1  ,"*",2  ,"*"],
                  ["*",1  ,"*",2  ],
                  [2  ,"*",1  ,"*"],
                  ["*",2  ,"*",1  ]]
    
    for i in range(len(f4_chain) - 1):
        if f4_chain[i] == f4_chain[i+1]:
            c3ot.append(0)
        else:
            c3ot.append(convertion[aux][f4_chain[i+1]])
            aux = f4_chain[i]
    c3ot.append(1)
    return c3ot

def contour(imagen):
    tamano = np.shape(imagen)
    white=1
    imagencontorno=np.zeros(tamano, dtype=int)
    for i in range(tamano[0]-1):
      for j in range(tamano[1]-1):
        if (i==0 or i==tamano[0]-1 or j==0 or j==tamano[1]-1):
          if (imagen[i][j]>=white):
            imagencontorno[i][j]=white
        else:
          if(imagen[i][j]>=white):
            if(imagen[i-1][j-1]<white):
              imagencontorno[i][j]=white
            else:
              if(imagen[i-1][j]<white):
                imagencontorno[i][j]=white
              else:
                if(imagen[i-1][j+1]<white):
                  imagencontorno[i][j]=white
                else:
                  if(imagen[i][j-1]<white):
                    imagencontorno[i][j]=white
                  else:
                    if(imagen[i][j+1]<white):
                      imagencontorno[i][j]=white
                    else:
                      if(imagen[i+1][j-1]<white):
                        imagencontorno[i][j]=white
                      else:
                        if(imagen[i+1][j]<white):
                          imagencontorno[i][j]=white
                        else:
                          if(imagen[i+1][j+1]<white):
                            imagencontorno[i][j]=white
    return imagencontorno

def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def show_f_chain_code(img, chain_code, coords):
    plt.imshow(img, cmap='gray')
    for i in range(len(chain_code)):
        plt.text(coords[i][1], coords[i][0], str(chain_code[i]), color='red', fontsize=8)
    plt.axis('off')
    plt.show()

def code_chain_to_str(chain_code):
    chain_code_str = ''.join(str(e) for e in chain_code)
    return chain_code_str

def preprocessing_image(img, dil:bool = True):
    if len(img.shape) > 2:
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        img = rgb2gray(img)
    if img.max() > 1:
        img_b = binarizar(img, 1, 0)
    else:
        img_b = img.astype(np.uint8)
    if dil:
        img_d = dilatation(img_b, 8)
    else:
        img_d = img_b
    return img_d

def f4(imgData:list, ret:bool = False):
    if len(imgData) == 3:
        chc_f4 = []
        img_f = imgData[2]
        for i in range(img_f.shape[2]):
            img_f4 = np.zeros((img_f[i].shape[0], img_f[i].shape[1], 2), dtype=np.uint8)
            for row, col in np.ndindex(img_f[i].shape):
                img_f4[row, col] = np.array([img_f[i][row, col], img_f[i][row, col]])
            if img_f4.max() == 1:
                chc_f4.append(freeman4ChainCode(img_f4))

    else:
        img = io.imread(imgData[1])
        img_f = contour(preprocessing_image(img))
        img_f4 = np.zeros((img_f.shape[0], img_f.shape[1], 2), dtype=np.uint8)
        for row, col in np.ndindex(img_f.shape):
            img_f4[row, col] = np.array([img_f[row, col], img_f[row, col]])
        chc_f4 = freeman4ChainCode(img_f4)

    if ret:
        return chc_f4

    if len(imgData) == 3:
        print(f"Codigos de cadena F4 de la imagen {imgData[0]}")
        for i, chc in enumerate(chc_f4):
            print(f"Capa {i+1}")
            print(code_chain_to_str(chc))
        input("Presione enter para continuar...")

    else:
        print(f"Codigo de cadena F4 de la imagen {imgData[0]}")
        print(code_chain_to_str(chc_f4))
        input("Presione enter para continuar...")

def f8(imgData:list, ret:bool = False, coords:bool = False):
    if len(imgData) == 3:
        chc_f8 = []
        coords_b = []
        img_f = imgData[2]
        for i in range(img_f.shape[2]):
            img_f[i] = preprocessing_image(img_f[i])
            if img_f[i].max() == 1:
                c, co = freeman8ChainCode(img_f[i])
                chc_f8.append(c)
                coords_b.append(co)
    else:
        img = io.imread(imgData[1])
        img_f = preprocessing_image(img)
        chc_f8, coords_b = freeman8ChainCode(img_f)

    if ret:
        if coords:
            return chc_f8, coords_b
        return chc_f8
    
    print(f"Codigo de cadena F8 de la imagen {imgData[0]}")
    if len(imgData) == 3:
        for i, chc in enumerate(chc_f8):
            print(f"Capa {i+1}")
            show_f_chain_code(img_f[i], chc, coords_b[i])
            print(code_chain_to_str(chc))
            if i != len(chc_f8) - 1:
                input("Presione enter para continuar...")
        
    else:
        show_f_chain_code(img_f, chc_f8, coords_b)
        print(code_chain_to_str(chc_f8))
    input("Presione enter para continuar...")

def vcc(imgData:list):
    f4_chain = f4(imgData, True)
    print(f"Codigo de cadena VCC de la imagen {imgData[0]}")
    if type(f4_chain[0]) == list:
        for i, chc in enumerate(f4_chain):
            print(f"Capa {i+1}")
            print(code_chain_to_str(VCC(chc)))
    else:
        vcc = VCC(f4_chain)
        print(code_chain_to_str(vcc))
    input("Presione enter para continuar...")

def threeOT(imgData:list):
    f4_chain = f4(imgData, True)
    print(f"Codigo de cadena 3OT de la imagen {imgData[0]}")
    if type(f4_chain[0]) == list:
        for i, chc in enumerate(f4_chain):
            print(f"Capa {i+1}")
            print(code_chain_to_str(c3OT(chc)))
    else:
        threeOT = c3OT(f4_chain)
        print(code_chain_to_str(threeOT))
    input("Presione enter para continuar...")

def af8(imgData:list, ret:bool = False, coords:bool = False):
    f8_chain, border_coords = f8(imgData, True, True)
    if type(f8_chain[0]) == list:
        af8 = []
        for i, chc in enumerate(f8_chain):
            af8.append(cAF8(chc))
    else:
        af8 = cAF8(f8_chain)
    if ret:
        if coords:
            return af8, border_coords
        return af8

    print(f"Codigo de cadena AF8 de la imagen {imgData[0]}")
    if type(f8_chain[0]) == list:
        for i, chc in enumerate(af8):
            print(f"Capa {i+1}")
            print(code_chain_to_str(chc))
            show_f_chain_code(preprocessing_image(imgData[2][i], dil=False), chc, border_coords[i])
            if i != len(f8_chain) - 1:
                input("Presione enter para continuar...")
    else:
        i = io.imread(imgData[1])
        show_f_chain_code(preprocessing_image(i), af8, border_coords)
        print(code_chain_to_str(af8))
    input("Presione enter para continuar...")
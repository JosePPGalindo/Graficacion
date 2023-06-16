from skimage import io
import numpy as np

class Euler2D: 
    def __init__(self, image) -> None:
        self.caracteristica_euler = 0
        self.uno_pixeles = np.sum(image)/255
        self.tetra_pixeles = 0
        self.vertices = 0
        self.aristas = 0
        self.hoyos = 0
        self.borde = 0 

class Euler3D: 
    def __init__(self, image) -> None:
        self.caracteristica_euler = 0
        self.uno_voxeles = np.sum(image)
        self.tetra_voxeles = 0
        self.vertices = 0
        self.aristas = 0
        self.tuneles = 0
        self.borde = 0 
        self.caras = 0
        self.octo_voxel = 0

class Unicos:
    def __init__(self):
        self.items = set()

    def add(self, item):
        normalized_item = tuple(sorted(map(tuple, item)))
        self.items.add(normalized_item)

    def calcula_total(self):
        return len([tuple(sorted(map(list, item))) for item in self.items])

def funcionesEuler(imgData:list):
    dos_de = False
    if len(imgData) == 3: 
        return imgData[2], dos_de, imgData[0]
    else: 
        dos_de = True
        img = io.imread(imgData[1])
        row, col = img.shape
        return [img, row, col], dos_de, imgData[0]

def caracteristicaEuler(imgData:list):
    imagen, dos_de, nombre = funcionesEuler(imgData)

    if dos_de:
        objeto_euler = calcEuler2D(image=imagen[0], row=imagen[1], column=imagen[2])
        print("Características de Euler: ", objeto_euler.caracteristica_euler)
        input("Presione enter para continuar...")
    else: 
        objeto_euler = calcularEuler3D(imagen)
        print("Características de Euler: ", objeto_euler.caracteristica_euler)
        input("Presione enter para continuar...")

    preguntar_imprimir(objeto_euler,nombre)

def calcularPixeles(imgData:list):
    imagen, dos_de, nombre = funcionesEuler(imgData)

    if dos_de:
        objeto_euler = calcEuler2D(image=imagen[0], row=imagen[1], column=imagen[2])
        print("Numero uno pixeles: ", objeto_euler.uno_pixeles)
        input("Presione enter para continuar...")
    else: 
        objeto_euler = calcularEuler3D(imagen)
        print("Numero uno voxeles: ", objeto_euler.uno_voxeles)
        input("Presione enter para continuar...")

    preguntar_imprimir(objeto_euler, nombre)

def calcularTetraPixeles(imgData:list):
    imagen, dos_de, nombre = funcionesEuler(imgData)

    if dos_de:
        objeto_euler = calcEuler2D(image=imagen[0], row=imagen[1], column=imagen[2])
        print("Numero tetra piixeles: ", objeto_euler.tetra_pixeles)
        input("Presione enter para continuar...")
    else: 
        objeto_euler = calcularEuler3D(imagen)
        print("Numero tetra voxeles: ", objeto_euler.tetra_voxeles)
        input("Presione enter para continuar...")

    preguntar_imprimir(objeto_euler, nombre)


def calcularVertices(imgData:list):
    imagen, dos_de, nombre = funcionesEuler(imgData)

    if dos_de:
        objeto_euler = calcEuler2D(image=imagen[0], row=imagen[1], column=imagen[2])
        print("Numero vertices: ", objeto_euler.vertices)
        input("Presione enter para continuar...")
    else: 
        objeto_euler = calcularEuler3D(imagen)
        print("Numero vertices: ", objeto_euler.vertices)
        input("Presione enter para continuar...")
    
    preguntar_imprimir(objeto_euler, nombre)

def calcularAristas(imgData:list):
    imagen, dos_de, nombre = funcionesEuler(imgData)

    if dos_de:
        objeto_euler = calcEuler2D(image=imagen[0], row=imagen[1], column=imagen[2])
        print("Numero aristas: ", objeto_euler.aristas)
        input("Presione enter para continuar...")
    else: 
        objeto_euler = calcularEuler3D(imagen)
        print("Numero aristas: ", objeto_euler.aristas)
        input("Presione enter para continuar...")
    
    preguntar_imprimir(objeto_euler, nombre)


def calcularHoyos(imgData:list):
    imagen, dos_de, nombre = funcionesEuler(imgData)

    if dos_de:
        objeto_euler = calcEuler2D(image=imagen[0], row=imagen[1], column=imagen[2])
        print("Numero de hoyos: ", objeto_euler.hoyos)
        input("Presione enter para continuar...")
    else: 
        objeto_euler = calcularEuler3D(imagen)
        print("Numero de tuneles: ", objeto_euler.tuneles)
        input("Presione enter para continuar...")
    preguntar_imprimir(objeto_euler, nombre)
    

def calcEuler2D(image, row, column):
    objeto_euler = Euler2D(image)
    relleno = np.pad(image, pad_width=1, mode='constant')
    lista_esquinas = Unicos()
    visitados = relleno.copy()

    vertices = np.zeros((row+1,column+1))
    for i in range(row):
        for j in range(column):
            if(visitados[i,j] == 255):
                vertices[i,j] = 1
                vertices[i,j+1] = 1
                vertices[i+1,j] = 1
                vertices[i+1,j+1] = 1   
                lista_esquinas.add(([i,j], [i,j+1]))
                lista_esquinas.add(([i,j], [i+1,j]))
                lista_esquinas.add(([i+1,j+1], [i+1,j]))
                lista_esquinas.add(([i+1,j+1], [i,j+1]))
                if(visitados[i,j-1] == 0):
                    objeto_euler.borde +=1
                if(visitados[i,j+1] == 0):
                    objeto_euler.borde +=1
                if(visitados[i+1,j] == 0):
                    objeto_euler.borde +=1
                if(visitados[i-1,j] == 0):
                    objeto_euler.borde +=1

    objeto_euler.vertices = np.sum(vertices)
    objeto_euler.aristas = lista_esquinas.calcula_total()
    objeto_euler.tetra_pixeles = objeto_euler.vertices - objeto_euler.borde
    objeto_euler.caracteristica_euler = objeto_euler.vertices - objeto_euler.aristas + objeto_euler.uno_pixeles

    return objeto_euler



def getcaras(visitados, z, y, x):
    superficie = 0
    caras = 6

    if(visitados[z-1,y,x] == 2):
        caras -=1
    if(visitados[z+1,y,x] == 2):
        caras -=1
    if(visitados[z,y-1,x] == 2):
        caras -=1
    if(visitados[z,y+1,x] == 2):
        caras -=1
    if(visitados[z,y,x-1] == 2):
        caras -=1
    if(visitados[z,y,x+1] == 2):
        caras -=1
    visitados[z, y, x] = 2
    if(visitados[z-1,y,x] == 0):
        superficie +=1
    if(visitados[z+1,y,x] == 0):
        superficie +=1
    if(visitados[z,y-1,x] == 0):
        superficie +=1
    if(visitados[z,y+1,x] == 0):
        superficie +=1
    if(visitados[z,y,x-1] == 0):
        superficie +=1
    if(visitados[z,y,x+1] == 0):
        superficie +=1

    return caras, superficie


def calcularEuler3D(image): 
    objeto_euler = Euler3D(image)
    
    padded_voxels = np.pad(image, pad_width=1, mode='constant')
    border_unicos = Unicos()
    superficie_area = 0

    visitados = padded_voxels.copy()

    vertices = np.zeros((visitados.shape[0]+1,visitados.shape[1]+1,visitados.shape[2]+1))
    for i in range(visitados.shape[0]):
        for j in range(visitados.shape[1]):
            for k in range(visitados.shape[2]):
                if(visitados[i,j,k] == 1):
                    vertices[i,j,k], vertices[i+1,j,k+1], vertices[i+1,j+1,k], vertices[i,j+1,k+1], vertices[i+1,j+1,k+1], vertices[i,j,k+1], vertices[i,j+1,k], vertices[i+1,j,k] = 1, 1, 1, 1, 1, 1, 1, 1   
                    guardar_bordes_unicos(i,j,k, border_unicos)

    objeto_euler.vertices = np.sum(vertices)
    objeto_euler.aristas = border_unicos.calcula_total()


    for i in range(visitados.shape[0]):
        for j in range(visitados.shape[1]):
            for k in range(visitados.shape[2]):
                if(visitados[i,j,k] == 1):
                    caras, superficie = getcaras(visitados, i, j, k)
                    objeto_euler.caras += caras
                    superficie_area += superficie

    objeto_euler.caracteristica_euler = objeto_euler.vertices-objeto_euler.aristas+objeto_euler.caras-objeto_euler.uno_voxeles
    objeto_euler.tetra_voxeles = objeto_euler.aristas - (2 * superficie_area)
    if(objeto_euler.caracteristica_euler <1):
        objeto_euler.tuneles = 1-objeto_euler.caracteristica_euler
    return objeto_euler

def guardar_bordes_unicos(i,j,k, border_unicos):
    border_unicos.add(([i,j,k], [i,j,k+1]))
    border_unicos.add(([i,j,k], [i,j+1,k]))
    border_unicos.add(([i,j,k], [i+1,j,k]))

    border_unicos.add(([i+1,j,k+1], [i+1,j,k]))
    border_unicos.add(([i+1,j,k+1], [i,j,k+1]))
    border_unicos.add(([i+1,j,k+1], [i+1,j+1,k+1]))

    border_unicos.add(([i+1,j+1,k], [i+1,j,k]))
    border_unicos.add(([i+1,j+1,k], [i,j+1,k]))
    border_unicos.add(([i+1,j+1,k], [i+1,j+1,k+1]))

    border_unicos.add(([i,j+1,k+1], [i,j,k+1]))
    border_unicos.add(([i,j+1,k+1], [i,j+1,k]))
    border_unicos.add(([i,j+1,k+1], [i+1,j+1,k+1]))

def preguntar_imprimir(objeto_euler, nombre): 
    print("¿Deseas guardar los datos calculados?")
    print("Tome en cuenta que al seleccionar sí, se guardarán todos los atributos de la imagen seleccionada anteriormente ")
    print("Estos datos son: Característica de Euler, número de voxeles o pixeles, número de hoyos o tuneles, número de tetra-pixeles o voxeles y número de vértices")
    opcion = input("1. Si\n2. No\n")
    if(opcion == "1"):
        hoyos = 0
        pixeles = 0 
        tetra_pixeles = 0
        try:
            hoyos = objeto_euler.hoyos
            pixeles = objeto_euler.uno_pixeles
            tetra_pixeles = objeto_euler.tetra_pixeles
        except: 
            hoyos = objeto_euler.tuneles
            pixeles = objeto_euler.uno_voxeles
            tetra_pixeles = objeto_euler.tetra_voxeles

        resultados = open("resultados_obtenidos.txt", "a")
        formato_de_resultado = f'''
        =================================================================================================================
        =================================== RESULTADOS DE IMAGEN {nombre} ===============================================
        =================================================================================================================
        \t\t\t\t\t Caracteristica de Euler: {objeto_euler.caracteristica_euler}
        \t\t\t\t\t Numero de hoyos o tuneles: {hoyos}
        \t\t\t\t\t Numero de voxeles o pixeles: {pixeles}
        \t\t\t\t\t Numero de tetra-pixeles o voxeles: {tetra_pixeles}
        \t\t\t\t\t Numero de vertices: {objeto_euler.vertices}
        =================================================================================================================\n\n
        '''
        resultados.write(formato_de_resultado)
        input("Los datos han sido guardados en resultados_obtenidos.txt, presione una tecla para continuar ")
    else:
        return
    pass

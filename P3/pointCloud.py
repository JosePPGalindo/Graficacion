from chainCodes import af8
import imageio
import matplotlib.pyplot as plt
from skimage import feature
from scipy.spatial.distance import euclidean

def convertir_a_letras(numeros):
    letras = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    resultado = []
    for numero in numeros:
        if numero >= 0 and numero < len(letras):
            resultado.append(letras[numero])
        else:
            resultado.append(str(numero)) 
    return resultado

def obtenerPuntosQuiebre(lista_letras):
    resultado = []
    i = 0

    while i < len(lista_letras):
        if lista_letras[i] == 'h' and i < len(lista_letras) - 1 and lista_letras[i+1] == 'b':
            resultado.append('h')
            resultado.append('b')
            i += 2
        elif lista_letras[i] == 'b' and i < len(lista_letras) - 1 and lista_letras[i+1] == 'h':
            resultado.append('b')
            resultado.append('h')
            i += 2
        elif lista_letras[i] == 'a':
            resultado.append('a')
            i += 1
        else:
            resultado.append('x')
            i += 1

    return resultado

def reducirCadena(lista):
    resultado = ""
    contador = 0

    for letra in lista:
        if letra == "a":
            contador += 1
        else:
            if contador > 0:
                resultado += "a^" + str(contador) + " "
                contador = 0
            resultado += letra

    if contador > 0:
        resultado += "a^" + str(contador) + " "

    return resultado

def obtener_borde_imagen(ruta_imagen, lista_letras):
    imagen = imageio.imread(ruta_imagen, as_gray=True)

    borde = feature.canny(imagen, sigma=1.0)

    for i in range(borde.shape[0]):
        for j in range(borde.shape[1]):
            if borde[i, j] != 0:
                if lista_letras[i] == 'x':
                    plt.scatter(i, j, color='red', s=10)
                else:
                    plt.scatter(i, j, color='blue', s=2)

    plt.axis("off")
    plt.show()

def printBreakPoints(coordenadas, lista, ret:bool = False):
    
    fig, ax = plt.subplots()
    
    distancia = []
    primero = False
    for i, coord in enumerate(coordenadas):
        x = coord[0]
        y = coord[1]
        
        if (lista[i] == 'x'):
            ax.scatter(x, y, s=50, color='red')
            if (primero == False):
                x_anterior = x
                y_anterior = y

                primero = True
            else:
                distancia.append(euclidean((x_anterior, y_anterior), (x, y)))
                x_anterior = x
                y_anterior = y
        else:
            ax.scatter(x, y, s=10, color='black')

    if ret:
        return distancia

    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

    return distancia

def freeContextGrammar(imageData:list, ret:bool = False):
    code, border_coords = af8(imageData, ret=True, coords=True)

    code = convertir_a_letras(code)

    grammar_complete = obtenerPuntosQuiebre(code)
    
    if ret:
        return border_coords, grammar_complete

    grammar_reduced = reducirCadena(grammar_complete)
    
    print(grammar_reduced)

    input("Presione 'Enter' para continuar")

def puntosQuiebre(imageData:list, ret:bool = False):
    border_coords, grammar_complete = freeContextGrammar(imageData, True)
    
    if ret == False:
        print(grammar_complete)
        preguntar_imprimir(grammar_complete, imageData[0])
    
    try:
        distancia = printBreakPoints(border_coords, grammar_complete, ret)
    except Exception as e:
        print("Ocurrió un error:", e)
        input("Presiona Enter para continuar...")

    if ret:
        return distancia
    input("Presione 'Enter' para continuar")

def integralSquareError(imageData:list):
    distancia = puntosQuiebre(imageData, True)

    ISE = 0
    for dist in distancia:
        ISE += dist ** 2

    print(f'ISE = {ISE}')
    input("Presione 'Enter' para continuar")

def preguntar_imprimir(datos, nombre): 
    print("¿Deseas guardar los datos calculados?")
    print("Tome en cuenta que al seleccionar sí, se guardarán todos los atributos de la imagen seleccionada anteriormente ")
    print("Estos datos son: Cadenas (gramática libre de contexto completa) ")
    opcion = input("1. Si\n2. No\n")
    if(opcion == "1"):
        resultados = open("resultados_obtenidos.txt", "a")
        formato_de_resultado = f'''
        =================================================================================================================
        =================================== RESULTADOS DE IMAGEN {nombre} ===============================================
        =================================================================================================================
        \t\t\t\t\t Cadenas (gramatica libre de contexto completa): {datos}
        =================================================================================================================\n\n
        '''
        resultados.write(formato_de_resultado)
        input("Los datos han sido guardados en resultados_obtenidos.txt, presione una tecla para continuar ")
    else:
        return
    pass

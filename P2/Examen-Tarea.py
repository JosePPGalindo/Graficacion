import cv2
import requests
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

urls = [
    "https://dabi.temple.edu/external/shape/MPEG7/original/apple-3.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/bottle-12.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/camel-2.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/chopper-04.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/classic-17.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/cup-4.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/device0-1.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/elephant-7.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/car-01.gif",
    "https://dabi.temple.edu/external/shape/MPEG7/original/hammer-8.gif",
]

images = []
pixels = []

def readImage(url:str):
    response = requests.get(url)
    img_arr = imageio.imread(response.content) / 255
    images.append(img_arr.astype(int))

def showImages(imgs):
    for image in imgs: 
      plt.figure(figsize=(image.shape[0] / 10,image.shape[1] / 11))
      plt.imshow(image, cmap='gray')
      plt.show()
 
for url in urls:
  readImage(url)

def calculate_centroid(image):
    height, width = image.shape
    area = np.sum(image)

    m10 = 0
    m01 = 0

    for y in range(height):
        for x in range(width):
            if image[y, x] == 1:
                m10 += x
                m01 += y

    centroid_x = m10 / area
    centroid_y = m01 / area

    return centroid_x, centroid_y


def calculate_central_moments(image, centroid_x, centroid_y, p, q):
    height, width = image.shape
    central_moment = 0

    for y in range(height):
        for x in range(width):
            if image[y, x] == 1:
                central_moment += ((x - centroid_x) ** p) * ((y - centroid_y) ** q)

    return central_moment


def calculate_normalized_moments(image):
    centroid_x, centroid_y = calculate_centroid(image)

    m00 = np.sum(image)

    mu20 = calculate_central_moments(image, centroid_x, centroid_y, 2, 0)
    mu02 = calculate_central_moments(image, centroid_x, centroid_y, 0, 2)
    mu11 = calculate_central_moments(image, centroid_x, centroid_y, 1, 1)
    mu30 = calculate_central_moments(image, centroid_x, centroid_y, 3, 0)
    mu03 = calculate_central_moments(image, centroid_x, centroid_y, 0, 3)
    mu21 = calculate_central_moments(image, centroid_x, centroid_y, 2, 1)
    mu12 = calculate_central_moments(image, centroid_x, centroid_y, 1, 2)

    nu20 = mu20 / (m00 ** (2 / 2 + 1))
    nu02 = mu02 / (m00 ** (2 / 2 + 1))
    nu11 = mu11 / (m00 ** (1 + 1))
    nu30 = mu30 / (m00 ** (3 / 2 + 1))
    nu03 = mu03 / (m00 ** (3 / 2 + 1))
    nu21 = mu21 / (m00 ** (2 + 1))
    nu12 = mu12 / (m00 ** (2 + 1))

    return [nu20, nu02, nu11, nu30, nu03, nu21, nu12]

def get_rotation_matrix_2d(center, angle, scale):
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    matrix = np.array([[cos_theta * scale, -sin_theta * scale, (1 - cos_theta) * center[0] + sin_theta * center[1]],
                       [sin_theta * scale, cos_theta * scale, -sin_theta * center[0] + (1 - cos_theta) * center[1]],
                       [0, 0, 1]])

    return matrix

def warp_affine(image, M, output_shape):
    height, width = output_shape
    output_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            p = np.array([x, y, 1])
            p_transformed = np.matmul(M, p)
            p_transformed = p_transformed.astype(int)

            if 0 <= p_transformed[0] < image.shape[1] and 0 <= p_transformed[1] < image.shape[0]:
                output_image[y, x] = image[p_transformed[1], p_transformed[0]]

    return output_image

imagenes = images

teta = 100
for imagen in imagenes:

    filas, columnas = imagen.shape[:2]
    matriz_rotacion = get_rotation_matrix_2d((columnas/2, filas/2), teta, 2)
    imagen_rotada = warp_affine(imagen, matriz_rotacion, (columnas, filas))

    hu_antes = calculate_normalized_moments(imagen)
    hu_despues = calculate_normalized_moments(imagen_rotada)
    
    plt.imshow(imagen_rotada)
    plt.show()
    print('Momentos de Hu antes de la rotación:', hu_antes)
    print('Momentos de Hu después de la rotación:', hu_despues)

input("Press enter to continue...")
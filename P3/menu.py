import os
from functions import open_img, save_info, open_3d_img
from chainCodes import f4, f8, vcc, threeOT, af8
from pointCloud import *
from eulerFunctions import caracteristicaEuler, calcularHoyos, calcularPixeles, calcularTetraPixeles, calcularVertices

def clean_console():
  os.system('cls' if os.name == 'nt' else 'clear')

def p_img_opts(opts):
  for index, opt in enumerate(opts, start=1):
    if index == 1:
      print("Imagenes 2D")
    elif index == 6:
      print("Imagenes 3D")
    print(f"{index}) {opt[0]}")

def get_img_options(opt:int = None):

  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  imgs = [
      ["Camello","imgs/2D/camel-2.jpg"],
      ["Auto Clasico","imgs/2D/classic-17.jpg"],
      ["Manzana","imgs/2D/apple-3.jpg"],
      ["Taza","imgs/2D/cup-4.jpg"],
      ["Martillo","imgs/2D/hammer-8.jpg"],
      ["Manzana","imgs/3D/apple.obj"],
      ["Football", "imgs/3D/football.obj"],
      ["Mango", "imgs/3D/mango.obj"],
      ["Orange", "imgs/3D/orange.obj"],
      ["Ojo","imgs/3D/eyeball.obj"],
  ]
  if opt:
    return imgs[opt]
  return imgs

def print_img_menu(func):
  while True:
    print("Menu")
    options = get_img_options()
    p_img_opts(options)
    print('S) Salir')
    select = input().lower()
    try:
      check = int(select)
      clean_console()
      if check > 5:
        op = options[check-1]
        op.append(open_3d_img(op, True))
        func(op)
      else:
        func(options[check-1])
      clean_console()
    except Exception:
      if select == 's':
        clean_console()
        return
      clean_console()


def get_main_opts(opt:int = None):
  m_opts = {
      1: {
          'title':'Funciones',
          'options': ['Abrir imagenes', 'Guardar la informacion obtenida'],
          'functions': [open_img, save_info]
      },
      2: {
          'title':'Codigos de cadena',
          'options': ['F4', 'F8', 'VCC', '3OT', 'AF8'],
          'functions': [f4, f8, vcc, threeOT, af8]
      },
      3: {
          'title':'Nubes de puntos',
          'options': ['GLC', 'Puntos de Quiebre', 'ISE'],
          'functions': [freeContextGrammar, puntosQuiebre, integralSquareError]
      },
      4: {
          'title':'Euler',
          'options': ['Caracteristica de euler','Numero de hoyos o tuneles', 'Calcular 1-pixeles o 1-voxeles', 'Calcular tetrapixeles', 'Calcular vertices'],
          'functions': [caracteristicaEuler, calcularHoyos, calcularPixeles, calcularTetraPixeles, calcularVertices]
      },
  }
  if opt:
    return m_opts[opt]
  return [(item[0], item[1]['title']) for item in m_opts.items()]

def print_menu(menu_data:dict = None, opt:str = ""):
  while True:
    if not menu_data:
      options = get_main_opts()
      print("Menu principal")
      for o in options:
        print(f'{o[0]}) {o[1]}')
      print('S) Salir')
      select = input().lower()
      try:
        check = int(select)
        clean_console()
        print_menu(get_main_opts(check))
      except Exception:
        if select == 's':
          print('Adios')
          return
        clean_console()
    else:
      print("Menu")
      print(menu_data['title'])
      for i, o in enumerate(menu_data['options']):
        print(f'{i+1}) {o}')
      print('S) Salir')
      select = input().lower()
      try:
        check = int(select)
        clean_console()
        print_img_menu(menu_data['functions'][check-1])
      except Exception:
        if select == 's':
          clean_console()
          return
        clean_console()
    
clean_console()
print_menu()
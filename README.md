# Graficación

Repositorio para almacenar el codigo del proyecto final de la materia de graficación

## Autores

- [@Pablo Emanuel Alvarez Naranjo](https://github.com/Pablo-Naranjo)
- [@Jose Pablo Padilla Galindo](https://github.com/JosePPGalindo)

## Instalación

### Instalar git

Seguir los pasos del 1 al 2 de esta [pagina](https://www.atlassian.com/es/git/tutorials/install-git#windows)

Esto te permitira tener git instalado para poder descargar facilmente los archivos de este repositorio

### Instalar python

Para poder ejecutar los codigos de este proyecto es necesario contar con la version 3.10 de python, la cual se puede instalar en la [pagina oficial](https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe)

Y seguir los pasos que están en esta [pagina](https://www.netveloper.com/instalar-python-en-windows) para instalar correctamente python

### Instalar VSCode
Una de las mejores formas de ver y ejecutar codigo es VSCode que cuenta con una gran variedad de extensiones y configuraciones de python, y la mejor noticia es que es un software gratuito.

Se puede descargar de su [página oficial](https://code.visualstudio.com/Download) y su instalación practicamente es darle clic al boton siguiente

#### Instalar la extension de python en VSCode (Opcional pero muy recomendable)

Al terminar la instalación de VSCode, ejecutar el software y abrir la seccion de extensiones que se encuentra en la barra de herramientas del lado izquierdo

![Screenshot Tuto Extension 1](https://github.com/gerardhios/Graficacion/blob/main/assets/extensionstuto1.png)

En la barra de busqueda escribir "python" y dar clic en el boton de instalar en el primer resultado

![Screenshot Tuto Extension 2](https://github.com/gerardhios/Graficacion/blob/main/assets/extensionstuto2.png)
## Obtener el código

Para poder correr los programas, se debe seguir la siguiente serie de instrucciones. Es importante que tengas el simbolo del sistema o CMD abierto

Ir a la carpeta de descargas

```bash
  cd Downloads
```

Clonar el repositorio

```bash
  git clone https://github.com/gerardhios/Graficacion.git
```

Ir a la carpeta del repositorio descargado
```bash
  cd Graficacion
```

Abrir VSCode

```bash
  code .
```

Crear un entorno virtual (Opcional)

Crear entornos virtuales de python permite tener instancias diferentes de librerias, para cada proyecto que se tenga, evitando problemas de versiones

Crear un entorno virtual es muy sencillo, teniendo python instalado y VSCode, el primer paso es abrir una terminal. La terminal se puede abrir en VSCode con las teclas ctrl + shift + ñ

Con la terminal abierta, ejecutar lo siguiente

```bash
  python -m venv env
```

El segundo paso es cambiar el entorno con el cual se ejecutaran los archivos de python, esto se realiza abriendo un archivo de python en VSCode, una vez abierto se da clic en la version de python (que es la que se instalo) esta se encuentra en la parte inferior derecha

![Screenshot Tuto Entorno Virtual 1](https://github.com/gerardhios/Graficacion/blob/main/assets/envtuto1.png)

Al dar clic se desplegara una pantalla donde podemos elegir el entorno de ejecucion, que por defecto es el de instalacion, pero se debe selecionar el del entorno virtual que se creo anteriormente. Este se distinguira por tener la leyenda "env"

![Screenshot Tuto Entorno Virtual 2](https://github.com/gerardhios/Graficacion/blob/main/assets/envtuto2.png)

Instalar las librerias necesarias
```bash
  pip install -r requirements.txt
```
## Ejecución del proyecto
Con VSCode abierto lo unico que falta es abrir con doble clic el archivo llamado "menu.py" y ejecutarlo con las teclas ctrl + F5

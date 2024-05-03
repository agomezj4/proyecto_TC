import sys
import os

# Añadir la carpeta 'src' al path para poder importar los módulos
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'src'))
from functions.orchestrating import run_processing

def main():
    run_processing()

if __name__ == "__main__":
    main()

import sys
import os

# Añadir la carpeta 'src' al path para poder importar los módulos
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'src'))
from functions.orchestrating import (run_processing,
                                     run_featuring)

def main():
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        if stage == 'all':
            print("Ejecutando todas las etapas del proyecto ...")
            run_processing()
            run_featuring()
        elif stage == 'preparation':
            run_processing()
        elif stage == 'feature_engineering':
            run_featuring()
        else:
            print(f"Etapa '{stage}' no reconocida. Las etapas válidas son: preparation")
    else:
        print("No se especificó una etapa. Uso: python __main__.py [etapa]")

if __name__ == "__main__":
    main()

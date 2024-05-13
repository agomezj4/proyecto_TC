import sys
import os

# Añadir la carpeta 'src' al path para poder importar los módulos
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'src'))
from functions.orchestration import (run_processing,
                                     run_featuring,
                                     run_model_input,
                                     run_models)

def main():
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        if stage == 'all pipelines':
            run_processing()
            run_featuring()
            run_model_input()
            run_models()

        elif stage == 'preparation pipeline':
            run_processing()

        elif stage == 'feature_engineering pipeline':
            run_featuring()

        elif stage == 'model_input pipeline':
            run_model_input()

        elif stage == 'models pipeline':
            run_models()

        else:
            print(f"Etapa '{stage}' no reconocida. Las etapas válidas son: preparation")
    else:
        print("No se especificó una etapa. Uso: python __main__.py [etapa]")

if __name__ == "__main__":
    main()

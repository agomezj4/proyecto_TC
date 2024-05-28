import sys
import os

# Añadir la carpeta 'src' al path para poder importar los módulos
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'src'))
from functions.orchestration import (run_processing,
                                     run_featuring,
                                     run_model_input,
                                     run_models,
                                     run_model_selection,
                                     run_predicting)

def main():
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        if stage == 'all_pipelines':
            run_processing()
            run_featuring()
            run_model_input()
            run_models()
            run_model_selection()
            run_predicting()

        elif stage == 'preparation_pipeline':
            run_processing()

        elif stage == 'feature_engineering_pipeline':
            run_featuring()

        elif stage == 'model_input_pipeline':
            run_model_input()

        elif stage == 'models_pipeline':
            run_models()

        elif stage == 'model_selection_pipeline':
            run_model_selection()

        elif stage == 'predicting_pipeline':
            run_predicting()

        else:
            print(f"Etapa '{stage}' no reconocida. Las etapas válidas son: preparation")
    else:
        print("No se especificó una etapa. Uso: python __main__.py [etapa]")


if __name__ == "__main__":
    main()

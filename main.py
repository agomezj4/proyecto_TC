import sys

# Añadir la carpeta 'src' al path para poder importar los módulos personalizados
sys.path.append('src/')

from orchestrator import run_processing, run_feature_engineering, train_model

def main():
    run_processing()
    run_feature_engineering()
    train_model()

if __name__ == "__main__":
    main()

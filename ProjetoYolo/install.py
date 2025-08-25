import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path


def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(" Python 3.8+ é necessário!")
        print(f"Versão atual: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f" Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def check_pip():
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                       check=True, capture_output=True)
        print(" pip - OK")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(" pip não encontrado!")
        return False


def install_requirements():
    print("\nInstalando dependências...")

    try:
        print("Atualizando pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                       check=True, capture_output=True)

        print("Instalando dependências do requirements.txt...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                                check=True, capture_output=True, text=True)

        print(" Dependências instaladas com sucesso!")
        return True

    except subprocess.CalledProcessError as e:
        print(f" Erro ao instalar dependências: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False


def download_yolo_model():
    print("\nBaixando modelo YOLO...")

    try:
        download_script = """
from ultralytics import YOLO
print("Baixando modelo YOLOv8n...")
model = YOLO('yolov8n.pt')
print("Modelo baixado com sucesso!")
"""

        result = subprocess.run([sys.executable, "-c", download_script],
                                check=True, capture_output=True, text=True)

        print(" Modelo YOLO baixado com sucesso!")
        return True

    except subprocess.CalledProcessError as e:
        print(f" Aviso: Não foi possível baixar o modelo YOLO automaticamente")
        print(f"O modelo será baixado na primeira execução")
        return False


def create_directories():
    print("\nCriando diretórios...")

    directories = [
        "outputs",
        "outputs/videos",
        "outputs/images",
        "outputs/logs",
        "data",
        "models"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f" Diretório criado: {directory}")


def create_sample_data():
    print("\nCriando dados de exemplo...")

    try:
        sample_config = {
            "test_mode": True,
            "sample_data": True,
            "created_by": "install.py"
        }

        import json
        with open("data/sample_config.json", "w") as f:
            json.dump(sample_config, f, indent=2)

        print(" Dados de exemplo criados")
        return True

    except Exception as e:
        print(f" Aviso: Não foi possível criar dados de exemplo: {e}")
        return False


def test_installation():
    print("\nTestando instalação...")

    try:
        test_script = """
try:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    print(" Imports principais - OK")

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    print(" OpenCV - OK")

    if torch.cuda.is_available():
        print(" PyTorch com CUDA - OK")
    else:
        print(" PyTorch CPU - OK")

except ImportError as e:
    print(f" Erro de import: {e}")
    sys.exit(1)
except Exception as e:
    print(f" Erro geral: {e}")
    sys.exit(1)
"""

        result = subprocess.run([sys.executable, "-c", test_script],
                                check=True, capture_output=True, text=True)

        print(" Teste de instalação - OK")
        return True

    except subprocess.CalledProcessError as e:
        print(f" Teste de instalação falhou: {e}")
        return False


def show_usage_instructions():
    print("\n" + "=" * 60)
    print(" INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 60)

    print("\n COMO USAR:")
    print("1. Executar com câmera:")
    print("   python main.py")

    print("\n2. Executar com vídeo:")
    print("   python main.py --video caminho/para/video.mp4")

    print("\n3. Configurar ROI:")
    print("   python main.py --roi-config")

    print("\n4. Executar demonstração:")
    print("   python demo.py")

    print("\n5. Executar testes:")
    print("   python test_system.py")

    print("\nCONTROLES:")
    print("• P: Pausar/Retomar")
    print("• R: Resetar sistema")
    print("• C: Configurar ROI")
    print("• S: Salvar estado atual")
    print("• Q: Sair")

    print("\nESTRUTURA CRIADA:")
    print("• outputs/: Resultados e relatórios")
    print("• data/: Dados de entrada")
    print("• models/: Modelos treinados")

    print("\nCONFIGURAÇÃO:")
    print("• Edite config.py para personalizar parâmetros")
    print("• Ajuste thresholds em config.py")
    print("• Modifique ROI_CONFIG para alterar região de interesse")

    print("\nAJUDA:")
    print("• Consulte README.md para documentação completa")
    print("• Execute python main.py --help para opções")
    print("• Verifique outputs/detection.log para logs")


def main():
    print(" INSTALADOR DO SISTEMA DE DETECÇÃO DE TÊNIS")
    print("=" * 50)

    if not check_python_version():
        sys.exit(1)

    if not check_pip():
        print(" Instale o pip primeiro!")
        sys.exit(1)

    if not install_requirements():
        print(" Falha na instalação das dependências!")
        sys.exit(1)

    download_yolo_model()

    create_directories()

    create_sample_data()

    if not test_installation():
        print(" Falha no teste de instalação!")
        sys.exit(1)

    show_usage_instructions()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstalação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nErro durante instalação: {e}")
        sys.exit(1)


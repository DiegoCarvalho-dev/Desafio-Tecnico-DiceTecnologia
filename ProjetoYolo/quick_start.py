import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path


def check_dependencies():
    print(" Verificando dependências...")

    required_modules = ['cv2', 'numpy', 'ultralytics']
    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
            print(f" {module} - OK")
        except ImportError:
            print(f" {module} - FALTANDO")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n Módulos faltando: {', '.join(missing_modules)}")
        print("Execute: python install.py")
        return False

    return True


def create_test_image():
    print("\nCriando imagem de teste...")

    img = np.zeros((480, 640, 3), dtype=np.uint8)

    img[:] = (100, 100, 100)

    cv2.rectangle(img, (0, 300), (640, 480), (50, 50, 50), -1)

    shoes = [
        {'pos': (100, 350), 'size': (60, 30), 'color': (0, 255, 0)},
        {'pos': (300, 380), 'size': (50, 25), 'color': (255, 0, 0)},
        {'pos': (500, 360), 'size': (55, 28), 'color': (0, 0, 255)},
    ]

    for i, shoe in enumerate(shoes):
        x, y = shoe['pos']
        w, h = shoe['size']
        color = shoe['color']

        cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, -1)
        cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 255, 255), 2)

        cv2.putText(img, f"Tenis {i + 1}", (x - 20, y - h // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    output_path = "data/test_image.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

    print(f" Imagem de teste criada: {output_path}")
    return output_path


def run_quick_demo():
    print("\nExecutando demonstração rápida...")

    try:
        from shoe_detector import ShoeDetector
        from roi_manager import ROIManager

        detector = ShoeDetector()
        roi_manager = ROIManager()

        roi_manager.set_roi([50, 250, 590, 450])

        test_image_path = "data/test_image.jpg"
        if not os.path.exists(test_image_path):
            test_image_path = create_test_image()

        image = cv2.imread(test_image_path)
        if image is None:
            print(" Erro ao carregar imagem de teste")
            return False

        print("Detectando tênis...")
        start_time = time.time()
        detections = detector.detect_shoes(image)
        detection_time = time.time() - start_time

        print(f"  Tempo de detecção: {detection_time:.3f}s")
        print(f" Tênis detectados: {len(detections)}")

        roi_detections = []
        for detection in detections:
            if roi_manager.is_bbox_in_roi(detection['bbox']):
                roi_detections.append(detection)

        print(f" Tênis na ROI: {len(roi_detections)}")

        result_image = image.copy()

        result_image = roi_manager.draw_roi(result_image)

        for detection in roi_detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']

            if confidence >= 0.8:
                color = (0, 255, 0)
            elif confidence >= 0.6:
                color = (255, 0, 0)
            else:
                color = (128, 128, 128)

            cv2.rectangle(result_image,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          color, 2)

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(result_image, label,
                        (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        info_text = [
            f"Tênis Detectados: {len(roi_detections)}",
            f"Tempo: {detection_time:.3f}s",
            f"Confiança Média: {np.mean([d['confidence'] for d in roi_detections]):.2f}" if roi_detections else "N/A"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(result_image, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25

        output_path = "outputs/quick_demo_result.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_image)

        print(f" Resultado salvo em: {output_path}")

        print("\nMostrando resultado...")
        print("Pressione qualquer tecla para fechar...")

        cv2.imshow('Demonstração Rápida - Resultado', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True

    except Exception as e:
        print(f"Erro na demonstração: {e}")
        return False


def show_menu():
    print("\n" + "=" * 50)
    print(" INÍCIO RÁPIDO - SISTEMA DE DETECÇÃO DE TÊNIS")
    print("=" * 50)

    print("\nEscolha uma opção:")
    print("1. Verificar dependências")
    print("2.  Criar imagem de teste")
    print("3. Executar demonstração rápida")
    print("4. Ver instruções completas")
    print("5. Sair")

    while True:
        try:
            choice = input("\nOpção (1-5): ").strip()

            if choice == '1':
                check_dependencies()

            elif choice == '2':
                create_test_image()

            elif choice == '3':
                if check_dependencies():
                    run_quick_demo()
                else:
                    print(" Dependências não estão instaladas!")
                    print("Execute: python install.py")

            elif choice == '4':
                show_instructions()

            elif choice == '5':
                print(" Até logo!")
                break

            else:
                print(" Opção inválida. Tente novamente.")

        except KeyboardInterrupt:
            print("\n\nAté logo!")
            break
        except Exception as e:
            print(f" Erro: {e}")


def show_instructions():
    print("\n" + "=" * 60)
    print(" INSTRUÇÕES COMPLETAS")
    print("=" * 60)

    print("\nINSTALAÇÃO:")
    print("1. Execute: python install.py")
    print("2. Aguarde a instalação das dependências")
    print("3. O modelo YOLO será baixado automaticamente")

    print("\nUSO BÁSICO:")
    print("1. Com câmera: python main.py")
    print("2. Com vídeo: python main.py --video video.mp4")
    print("3. Configurar ROI: python main.py --roi-config")

    print("\nCONTROLES:")
    print("• P: Pausar/Retomar")
    print("• R: Resetar sistema")
    print("• C: Configurar ROI")
    print("• S: Salvar estado")
    print("• Q: Sair")

    print("\nESTRUTURA:")
    print("• main.py: Sistema principal")
    print("• demo.py: Demonstração interativa")
    print("• test_system.py: Testes automatizados")
    print("• config.py: Configurações")

    print("\nAJUDA:")
    print("• README.md: Documentação completa")
    print("• python main.py --help: Opções de linha de comando")
    print("• outputs/detection.log: Logs do sistema")


def main():
    print(" SISTEMA DE DETECÇÃO E CONTAGEM DE TÊNIS NO CHÃO")
    print("Script de Início Rápido")

    if not os.path.exists("main.py"):
        print(" Execute este script no diretório do projeto!")
        print("cd Projeto-vaga")
        print("python quick_start.py")
        return

    show_menu()


if __name__ == "__main__":
    main()

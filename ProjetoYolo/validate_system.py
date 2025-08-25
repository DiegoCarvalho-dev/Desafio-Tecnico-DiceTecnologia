import os
import sys
import time
import json
import subprocess
from pathlib import Path


class SystemValidator:
    def __init__(self):
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {},
            'validation_results': {},
            'overall_score': 0,
            'status': 'UNKNOWN'
        }

        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def validate_python_environment(self):
        print(" Validando ambiente Python...")

        try:
            version = sys.version_info
            self.results['system_info']['python_version'] = f"{version.major}.{version.minor}.{version.micro}"

            if version.major < 3 or (version.major == 3 and version.minor < 8):
                print(" Python 3.8+ é necessário!")
                return False

            print(f" Python {version.major}.{version.minor}.{version.micro} - OK")

            try:
                subprocess.run([sys.executable, "-m", "pip", "--version"],
                               check=True, capture_output=True)
                print(" pip - OK")
            except:
                print(" pip não encontrado!")
                return False

            return True

        except Exception as e:
            print(f" Erro na validação Python: {e}")
            return False

    def validate_dependencies(self):
        print("\nValidando dependências...")

        required_modules = [
            'cv2', 'numpy', 'torch', 'torchvision', 'ultralytics',
            'matplotlib', 'seaborn', 'PIL', 'sklearn'
        ]

        missing_modules = []
        available_modules = []

        for module in required_modules:
            try:
                __import__(module)
                available_modules.append(module)
                print(f" {module} - OK")
            except ImportError:
                missing_modules.append(module)
                print(f" {module} - FALTANDO")

        self.results['validation_results']['dependencies'] = {
            'available': available_modules,
            'missing': missing_modules,
            'status': 'PASS' if not missing_modules else 'FAIL'
        }

        if missing_modules:
            print(f"\nMódulos faltando: {', '.join(missing_modules)}")
            print("Execute: python install.py")
            return False

        return True

    def validate_file_structure(self):
        print("\nValidando estrutura de arquivos...")

        required_files = [
            'main.py', 'shoe_detector.py', 'roi_manager.py', 'event_tracker.py',
            'utils.py', 'config.py', 'requirements.txt', 'README.md'
        ]

        required_dirs = ['data', 'outputs', 'models']

        missing_files = []
        missing_dirs = []

        for file in required_files:
            if os.path.exists(file):
                print(f" {file} - OK")
            else:
                missing_files.append(file)
                print(f" {file} - FALTANDO")

        for directory in required_dirs:
            if os.path.exists(directory):
                print(f" {directory}/ - OK")
            else:
                missing_dirs.append(directory)
                print(f" {directory}/ - FALTANDO")

        self.results['validation_results']['file_structure'] = {
            'missing_files': missing_files,
            'missing_directories': missing_dirs,
            'status': 'PASS' if not missing_files and not missing_dirs else 'FAIL'
        }

        return len(missing_files) == 0 and len(missing_dirs) == 0

    def validate_imports(self):
        print("\nValidando imports...")

        import_tests = [
            ('shoe_detector', 'ShoeDetector'),
            ('roi_manager', 'ROIManager'),
            ('event_tracker', 'EventTracker'),
            ('utils', 'setup_logging'),
            ('config', 'MODEL_CONFIG')
        ]

        failed_imports = []
        successful_imports = []

        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name)
                if hasattr(module, class_name):
                    successful_imports.append(f"{module_name}.{class_name}")
                    print(f" {module_name}.{class_name} - OK")
                else:
                    failed_imports.append(f"{module_name}.{class_name}")
                    print(f" {module_name}.{class_name} - FALTANDO")
            except ImportError as e:
                failed_imports.append(f"{module_name}.{class_name}")
                print(f" {module_name}.{class_name} - ERRO: {e}")

        self.results['validation_results']['imports'] = {
            'successful': successful_imports,
            'failed': failed_imports,
            'status': 'PASS' if not failed_imports else 'FAIL'
        }

        return len(failed_imports) == 0

    def validate_yolo_model(self):
        print("\nValidando modelo YOLO...")

        try:
            from ultralytics import YOLO

            start_time = time.time()
            model = YOLO('yolov8n.pt')
            load_time = time.time() - start_time

            print(f" Modelo YOLOv8n carregado em {load_time:.2f}s")
            print(f" Dispositivo: {model.device}")

            self.results['validation_results']['yolo_model'] = {
                'status': 'PASS',
                'model_name': 'yolov8n.pt',
                'device': str(model.device),
                'load_time': load_time
            }

            return True

        except Exception as e:
            print(f" Erro ao carregar modelo YOLO: {e}")

            self.results['validation_results']['yolo_model'] = {
                'status': 'FAIL',
                'error': str(e)
            }

            return False

    def validate_basic_functionality(self):
        print("\nValidando funcionalidades básicas...")

        tests = []

        try:
            from shoe_detector import ShoeDetector
            detector = ShoeDetector()
            tests.append(('Criação do detector', True))
            print(" Criação do detector - OK")

            from roi_manager import ROIManager
            roi_manager = ROIManager()
            tests.append(('Criação do ROI manager', True))
            print(" Criação do ROI manager - OK")

            from event_tracker import EventTracker
            event_tracker = EventTracker()
            tests.append(('Criação do event tracker', True))
            print(" Criação do event tracker - OK")

            test_roi = [50, 50, 590, 430]
            roi_manager.set_roi(test_roi)
            tests.append(('Configuração de ROI', True))
            print(" Configuração de ROI - OK")

            current_roi = roi_manager.get_roi()
            if current_roi == test_roi:
                tests.append(('Verificação de ROI', True))
                print(" Verificação de ROI - OK")
            else:
                tests.append(('Verificação de ROI', False))
                print(" Verificação de ROI - FALHOU")

        except Exception as e:
            print(f" Erro nos testes básicos: {e}")
            tests.append(('Testes básicos', False))

        passed = sum(1 for test, result in tests if result)
        total = len(tests)

        self.results['validation_results']['basic_functionality'] = {
            'tests': tests,
            'passed': passed,
            'total': total,
            'status': 'PASS' if passed == total else 'FAIL'
        }

        return passed == total

    def validate_performance(self):
        print("\nValidando performance...")

        try:
            import cv2
            import numpy as np

            test_image = np.zeros((480, 640, 3), dtype=np.uint8)

            start_time = time.time()
            resized = cv2.resize(test_image, (320, 240))
            resize_time = time.time() - start_time

            start_time = time.time()
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            operations_time = time.time() - start_time

            print(f" Redimensionamento: {resize_time * 1000:.2f}ms")
            print(f" Operações básicas: {operations_time * 1000:.2f}ms")

            self.results['validation_results']['performance'] = {
                'status': 'PASS',
                'resize_time_ms': resize_time * 1000,
                'operations_time_ms': operations_time * 1000
            }

            return True

        except Exception as e:
            print(f" Erro nos testes de performance: {e}")

            self.results['validation_results']['performance'] = {
                'status': 'FAIL',
                'error': str(e)
            }

            return False

    def run_unit_tests(self):

        print("\nExecutando testes unitários...")

        try:
            result = subprocess.run([sys.executable, "test_system.py"],
                                    capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print(" Testes unitários - PASS")
                self.results['validation_results']['unit_tests'] = {
                    'status': 'PASS',
                    'output': result.stdout
                }
                return True
            else:
                print(" Testes unitários - FAIL")
                print(f"Erro: {result.stderr}")

                self.results['validation_results']['unit_tests'] = {
                    'status': 'FAIL',
                    'error': result.stderr,
                    'output': result.stdout
                }
                return False

        except subprocess.TimeoutExpired:
            print(" Testes unitários - TIMEOUT")
            self.results['validation_results']['unit_tests'] = {
                'status': 'FAIL',
                'error': 'Timeout após 60 segundos'
            }
            return False
        except Exception as e:
            print(f" Erro ao executar testes: {e}")
            self.results['validation_results']['unit_tests'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False

    def calculate_overall_score(self):
        print("\nCalculando pontuação geral...")

        validation_results = self.results['validation_results']
        total_checks = len(validation_results)
        passed_checks = 0

        for check_name, check_result in validation_results.items():
            if check_result.get('status') == 'PASS':
                passed_checks += 1

        score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

        self.results['overall_score'] = score

        if score >= 90:
            status = 'EXCELLENT'
        elif score >= 80:
            status = 'GOOD'
        elif score >= 70:
            status = 'FAIR'
        elif score >= 60:
            status = 'POOR'
        else:
            status = 'FAIL'

        self.results['status'] = status

        print(f" Pontuação geral: {score:.1f}% ({status})")
        print(f" Verificações passadas: {passed_checks}/{total_checks}")

        return score

    def save_results(self):
        output_path = "outputs/validation_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)

            print(f"\nResultados salvos em: {output_path}")

        except Exception as e:
            print(f" Erro ao salvar resultados: {e}")

    def show_summary(self):
        print("\n" + "=" * 60)
        print("RESUMO DA VALIDAÇÃO")
        print("=" * 60)

        print(f" Data/Hora: {self.results['timestamp']}")
        print(f" Python: {self.results['system_info'].get('python_version', 'N/A')}")
        print(f" Status Geral: {self.results['status']}")
        print(f" Pontuação: {self.results['overall_score']:.1f}%")

        print("\nDETALHES:")
        for check_name, check_result in self.results['validation_results'].items():
            status = check_result.get('status', 'UNKNOWN')
            status_icon = "✅" if status == 'PASS' else "❌"
            print(f"{status_icon} {check_name}: {status}")

        print("\n" + "=" * 60)

        if self.results['status'] in ['EXCELLENT', 'GOOD']:
            print(" SISTEMA VALIDADO COM SUCESSO!")
            print(" Todas as funcionalidades principais estão funcionando")
        elif self.results['status'] == 'FAIR':
            print("  SISTEMA PARCIALMENTE FUNCIONAL")
            print(" Algumas funcionalidades podem precisar de ajustes")
        else:
            print(" SISTEMA COM PROBLEMAS")
            print(" Verifique os erros e execute python install.py")

    def run_full_validation(self):
        print(" INICIANDO VALIDAÇÃO COMPLETA DO SISTEMA")
        print("=" * 50)

        validations = [
            ('Ambiente Python', self.validate_python_environment),
            ('Dependências', self.validate_dependencies),
            ('Estrutura de Arquivos', self.validate_file_structure),
            ('Imports', self.validate_imports),
            ('Modelo YOLO', self.validate_yolo_model),
            ('Funcionalidades Básicas', self.validate_basic_functionality),
            ('Performance', self.validate_performance),
            ('Testes Unitários', self.run_unit_tests)
        ]

        for validation_name, validation_func in validations:
            print(f"\n{validation_name}...")
            try:
                if validation_func():
                    print(f" {validation_name} - VALIDADO")
                else:
                    print(f" {validation_name} - FALHOU")
            except Exception as e:
                print(f" {validation_name} - ERRO: {e}")

        self.calculate_overall_score()
        self.save_results()
        self.show_summary()


def main():
    try:
        validator = SystemValidator()
        validator.run_full_validation()

    except KeyboardInterrupt:
        print("\n\nValidação interrompida pelo usuário")
    except Exception as e:
        print(f"\n\nErro durante validação: {e}")


if __name__ == "__main__":
    main()


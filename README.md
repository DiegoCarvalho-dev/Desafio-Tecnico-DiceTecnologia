# Sistema de Detecção e Contagem de Tênis no Chão

## Descrição
Sistema de visão computacional que detecta, conta e monitora tênis em uma área definida, simulando um processo de coleta manual com câmera fixa.

## Funcionalidades
- ✅ Detecção automática de tênis usando YOLO v8
- ✅ Definição de ROI (Região de Interesse) para o chão
- ✅ Contagem em tempo real de tênis visíveis
- ✅ Detecção de eventos de coleta (tênis removidos)
- ✅ Relatório final com estatísticas completas
- ✅ Interface visual com tracking de objetos

## Instalação

### 1. Clone o repositório
```bash
git clone (https://github.com/DiegoCarvalho-dev/Desafio-Tecnico-DiceTecnologia.git)
cd Projeto-vaga
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Baixe o modelo YOLO (opcional - será baixado automaticamente)
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Uso

### Execução Principal
```bash
python main.py
```

### Execução com Vídeo Personalizado
```bash
python main.py --video caminho/para/video.mp4
```

### Execução com Câmera
```bash
python main.py --camera 0
```

### Configuração de ROI
```bash
python main.py --roi-config
```

## Estrutura do Projeto
```
Projeto-vaga/
├── main.py                 # Script principal
├── shoe_detector.py        # Classe principal de detecção
├── roi_manager.py          # Gerenciamento de ROI
├── event_tracker.py        # Rastreamento de eventos
├── utils.py                # Funções utilitárias
├── config.py               # Configurações do sistema
├── requirements.txt        # Dependências
├── README.md              # Documentação
├── data/                  # Dados de teste
├── models/                # Modelos treinados
└── outputs/               # Resultados e relatórios
```

## Arquitetura Técnica

### Componentes Principais
1. **ShoeDetector**: Classe principal que integra YOLO com lógica de tracking
2. **ROIManager**: Gerencia a região de interesse e coordenadas do chão
3. **EventTracker**: Monitora eventos de coleta e movimentação
4. **VideoProcessor**: Processa vídeo/câmera em tempo real

### Fluxo de Processamento
1. Captura de frame da câmera/vídeo
2. Detecção de objetos usando YOLO
3. Filtragem por classe "shoe" e confiança
4. Aplicação de ROI para limitar área de detecção
5. Tracking de objetos entre frames
6. Detecção de eventos de coleta
7. Geração de estatísticas e relatórios

## Configurações

### Parâmetros Ajustáveis
- `CONFIDENCE_THRESHOLD`: Limite de confiança para detecção (0.5)
- `ROI_MARGIN`: Margem da ROI em pixels (50)
- `TRACKING_THRESHOLD`: Limite para considerar objeto como "coletado" (0.7)
- `MODEL_SIZE`: Tamanho do modelo YOLO ('n', 's', 'm', 'l', 'x')

### Personalização de ROI
Execute com `--roi-config` para definir manualmente a região de interesse.

## Saídas

### Relatório Final
- Total de tênis detectados
- Quantidade coletada
- Quantidade restante
- Tempo de processamento
- Estatísticas de confiança

### Arquivos Gerados
- `outputs/detection_results.json`: Resultados detalhados
- `outputs/processed_video.mp4`: Vídeo processado (se aplicável)
- `outputs/roi_config.json`: Configuração da ROI

## Performance

### Métricas Esperadas
- **FPS**: 15-30 (dependendo do hardware)
- **Precisão**: 85-95% (dependendo da qualidade da imagem)
- **Latência**: <100ms para detecção

### Otimizações
- Processamento em GPU (se disponível)
- Redimensionamento de frames para melhor performance
- Filtragem inteligente de detecções

## Troubleshooting

### Problemas Comuns
1. **Modelo não baixa**: Verifique conexão com internet
2. **Performance baixa**: Reduza resolução ou use modelo menor
3. **Detecções incorretas**: Ajuste threshold de confiança
4. **ROI não funciona**: Execute configuração manual

### Logs
O sistema gera logs detalhados para debug. Verifique o console para mensagens de erro.

## Contribuição
Para contribuir com o projeto:
1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Abra um Pull Request

## Contato
Para dúvidas ou suporte, abra uma issue no repositório.

Ass: Diego Ricado Carvalho 

# 🚀 INSTRUÇÕES RÁPIDAS - SISTEMA DE DETECÇÃO DE TÊNIS

## ⚡ INÍCIO SUPER RÁPIDO (5 minutos)

### 1. 🔧 INSTALAÇÃO AUTOMÁTICA
```bash
# Execute o instalador automático
python install.py
```

### 2. 🧪 TESTE RÁPIDO
```bash
# Execute o script de início rápido
python quick_start.py
```

### 3. 🎯 PRIMEIRA EXECUÇÃO
```bash
# Execute com câmera (padrão)
python main.py
```

## 📋 COMANDOS ESSENCIAIS

| Comando | Descrição |
|---------|-----------|
| `python main.py` | Sistema com câmera |
| `python main.py --video video.mp4` | Processar vídeo |
| `python main.py --roi-config` | Configurar ROI |
| `python demo.py` | Demonstração interativa |
| `python test_system.py` | Testes automatizados |
| `python validate_system.py` | Validação completa |

## 🎮 CONTROLES EM TEMPO REAL

- **P**: Pausar/Retomar
- **R**: Resetar sistema
- **C**: Configurar ROI
- **S**: Salvar estado
- **Q**: Sair

## 🔧 CONFIGURAÇÃO RÁPIDA

### Ajustar Sensibilidade
Edite `config.py`:
```python
MODEL_CONFIG = {
    'confidence_threshold': 0.5,  # Ajustar aqui (0.1 a 1.0)
    'iou_threshold': 0.45,        # Ajustar aqui (0.1 a 1.0)
}
```

### Ajustar ROI
```python
ROI_CONFIG = {
    'margin': 50,  # Margem da ROI em pixels
    'auto_detect': True,  # False para manual
}
```

## 📊 INTERPRETAÇÃO DOS RESULTADOS

### Interface Visual
- **Verde**: Tênis detectados com alta confiança
- **Azul**: Tênis em tracking
- **Vermelho**: Tênis coletados
- **Retângulo verde**: ROI (Região de Interesse)

### Estatísticas
- **Tênis Detectados**: Total atual na ROI
- **Total Coletados**: Quantidade removida
- **FPS**: Performance do sistema
- **Confiança**: Precisão da detecção

## 🚨 SOLUÇÃO DE PROBLEMAS RÁPIDA

### ❌ "Modelo não carrega"
```bash
python install.py
# Aguarde download automático
```

### ❌ "Câmera não funciona"
```bash
python main.py --camera 1  # Tentar câmera 1
python main.py --camera 2  # Tentar câmera 2
```

### ❌ "Performance baixa"
1. Edite `config.py`
2. Mude `model_size` para `'n'` (nano)
3. Reduza `frame_width` e `frame_height`

### ❌ "Detecções incorretas"
1. Ajuste `confidence_threshold` em `config.py`
2. Configure ROI manualmente: `python main.py --roi-config`
3. Verifique iluminação da cena

## 📁 ESTRUTURA DE ARQUIVOS

```
Projeto-vaga/
├── main.py              # 🎯 SISTEMA PRINCIPAL
├── shoe_detector.py     # 🤖 Detector YOLO
├── roi_manager.py       # 📐 Gerenciador de ROI
├── event_tracker.py     # 📊 Rastreador de eventos
├── utils.py             # 🛠️ Funções utilitárias
├── config.py            # ⚙️ Configurações
├── demo.py              # 🎪 Demonstração
├── test_system.py       # 🧪 Testes
├── install.py           # 🔧 Instalador
├── quick_start.py       # ⚡ Início rápido
├── validate_system.py   # ✅ Validador
├── requirements.txt     # 📦 Dependências
├── README.md            # 📖 Documentação
├── data/                # 📊 Dados de teste
├── outputs/             # 💾 Resultados
└── models/              # 🤖 Modelos YOLO
```

## 🎯 CENÁRIOS DE USO

### 🏠 **Teste em Casa**
```bash
python quick_start.py
# Opção 3: Executar demonstração rápida
```

### 🎥 **Processar Vídeo**
```bash
python main.py --video caminho/para/video.mp4
```

### 📹 **Streaming com Câmera**
```bash
python main.py
# Use controles P, R, C, S, Q
```

### 🔬 **Desenvolvimento/Testes**
```bash
python test_system.py
python validate_system.py
```

## 📈 OTIMIZAÇÃO DE PERFORMANCE

### Para **Alta Velocidade**:
```python
# config.py
MODEL_CONFIG = {
    'model_size': 'n',  # Nano (mais rápido)
    'confidence_threshold': 0.7,  # Menos detecções
}
VIDEO_CONFIG = {
    'frame_width': 320,   # Resolução menor
    'frame_height': 240,
}
```

### Para **Alta Precisão**:
```python
# config.py
MODEL_CONFIG = {
    'model_size': 'm',  # Medium (mais preciso)
    'confidence_threshold': 0.3,  # Mais detecções
}
VIDEO_CONFIG = {
    'frame_width': 1280,  # Resolução maior
    'frame_height': 720,
}
```

## 🆘 AJUDA ADICIONAL

### 📖 Documentação Completa
- `README.md` - Documentação detalhada
- `config.py` - Todas as configurações
- Comentários no código

### 🧪 Validação do Sistema
```bash
python validate_system.py
# Executa testes completos e gera relatório
```

### 📊 Logs e Debug
- Logs salvos em: `outputs/detection.log`
- Resultados em: `outputs/`
- Configurações em: `outputs/roi_config.json`

## 🎉 PRONTO PARA USAR!

O sistema está configurado para funcionar imediatamente após a instalação. Use `python quick_start.py` para uma demonstração rápida ou `python main.py` para começar a usar com sua câmera!

---

**💡 DICA**: Execute `python validate_system.py` para verificar se tudo está funcionando perfeitamente!




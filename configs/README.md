# `configs/` — presets de treino (YAML)

A **fonte única de hiperparâmetros**: os scripts não têm nada *hardcoded*, só
consomem o que sai daqui (`tucanoce.config.load_configs`). Cada YAML tem três
blocos: `model` (arquitetura/preset), `train` (otimização) e `data` (caminhos de
corpus, tokenizer e cache).

| Config | Uso |
|---|---|
| `small.yaml` | Preset dev/CPU mínimo; roda de ponta a ponta com o corpus *fallback* embutido. |
| `tinystories_cpu.yaml` | Treino no TinyStories em CPU (**recomendado** p/ o modelo pequeno). |
| `ml_cpu.yaml` | Treino no corpus de *machine learning* (Wikipedia) em CPU. |
| `physics_cpu.yaml` | Treino no corpus de física de partículas em CPU (experimento original). |
| `medium.yaml` | Preset "carro-chefe" (~43M params) para GPU. |

Os presets de arquitetura (`small`→`xl`) e o cálculo de `hidden_dim` do SwiGLU
vivem em [`../src/tucanoce/config.py`](../src/tucanoce/config.py). Trocar de corpus é
trocar os caminhos no bloco `data:` (e usar um `tokenizer_path`/`tokens_cache`
próprios para o cache invalidar corretamente).

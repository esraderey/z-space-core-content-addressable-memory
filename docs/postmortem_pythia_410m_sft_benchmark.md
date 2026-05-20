# Postmortem: Pythia-410M SFT Checkpoint Benchmark

Fecha: 2026-05-14

## Objetivo

Benchmark solicitado:

- Modelo: `EleutherAI/pythia-410m`
- Dataset: WikiText `wikitext-103-raw-v1`
- Entrenamiento: SFT real
- Checkpoints: 100
- Intervalo: cada 50 steps
- Metricas: growth por checkpoint, tiempo de commit y tiempo de reconstruccion final

## Entorno

- GPU: NVIDIA GeForce RTX 4060, 8 GB VRAM
- PyTorch: `2.11.0+cu128`
- CUDA visible desde PyTorch: si
- Dependencias instaladas: `transformers`, `datasets`, `accelerate`, `safetensors`, `huggingface_hub`
- Dtype usado: `bf16`
- Batch size: 1
- Block size de secuencia: 512
- Gradient checkpointing: activado
- Store usado en la corrida historica: `DiskContentStore` en `scripts/benchmark_sft_pythia_410m.py`
- Store actual del runner corregido: `PackfileContentStore` append-only en `.zspace_bench/pythia_410m_wikitext/store`

## Estado Del Fix P0

Despues de esta auditoria se corrigieron los tres P0 de implementacion:

- Versionado de checkpoints con delta denso lossless `XOR+zstd` por tensor.
- Fast path para `decomp_type=RAW`, sin compresion de scoring redundante.
- Store append-only tipo packfile para evitar miles de archivos por checkpoint.

El benchmark ahora permite comparar ambos caminos:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_sft_pythia_410m.py --checkpoint-mode full
.\.venv\Scripts\python.exe scripts\benchmark_sft_pythia_410m.py --checkpoint-mode xor-zstd --zstd-level 3
```

El runner tambien fija por defecto los caches de Hugging Face dentro del
workspace para evitar errores de permisos en `C:\Users\Raul\.cache`:

```powershell
--hf-home .zspace_bench/hf_home
--hf-datasets-cache .zspace_bench/hf_datasets
```

## Estado De La Corrida

La corrida completa fue detenida manualmente porque la tendencia era concluyente y no iba a mejorar con mas checkpoints.

Ultimo checkpoint registrado:

- Checkpoint: 27 / 100
- Step: 1350 / 5000
- Store Z-Space: `21,675,513,579` bytes
- Checkpoints PyTorch equivalentes acumulados: `21,890,682,513` bytes
- Ratio Z-Space / PyTorch acumulado: `0.99017`
- Reconstruccion final del benchmark completo: no ejecutada, porque la corrida se detuvo antes del checkpoint 100
- Validacion previa minima: una corrida de 1 checkpoint reconstruyo exacto (`final_exact=true`)

## Evidencia

Resumen agregado de los 27 checkpoints registrados:

| Metrica | Valor |
|---|---:|
| Checkpoints registrados | 27 |
| Growth total Z-Space | 21,675,513,579 bytes |
| Growth promedio | 802,796,799 bytes |
| Growth minimo | 800,421,171 bytes |
| Growth maximo | 810,064,531 bytes |
| Commit promedio Z-Space | 91.27 s |
| Commit minimo Z-Space | 82.60 s |
| Commit maximo Z-Space | 105.14 s |
| Checkpoint PyTorch promedio | 810,766,019 bytes |
| `torch.save` promedio | 3.62 s |
| Ratio acumulado Z-Space / PyTorch | 99.02% |

Ultimos checkpoints:

| Checkpoint | Step | Growth Z-Space | Store Z-Space | PyTorch checkpoint | Commit Z-Space |
|---:|---:|---:|---:|---:|---:|
| 25 | 1250 | 802,825,900 | 20,070,188,239 | 810,766,019 | 96.55 s |
| 26 | 1300 | 803,938,198 | 20,874,126,437 | 810,766,019 | 100.38 s |
| 27 | 1350 | 801,387,142 | 21,675,513,579 | 810,766,019 | 105.14 s |

## Addendum: Medicion De Delta Lossless Entre Checkpoints Consecutivos

Despues de la primera auditoria se ejecuto una medicion adicional con dos checkpoints consecutivos reales:

- `c_i`: checkpoint en step 50
- `c_{i+1}`: checkpoint en step 100
- Modelo/dataset/config: igual al benchmark original
- Metodo 1: XOR de bytes tensoriales canonicos alineados por clave y `zstd(level=3)`
- Metodo 2: XOR de archivos `torch.save` completos y `zstd(level=3)`

Resultados:

| Medicion | Bytes Base | `zstd(XOR)` | Ratio |
|---|---:|---:|---:|
| Tensorial canonico alineado | 810,668,032 | 69,015,298 | 8.51% |
| Archivo `torch.save` | 810,763,273 | 69,079,070 | 8.52% |

Este resultado invalida la conclusion fuerte de que "hay que abandonar lossless". El bound practico medido no esta cerca de 700 MB; esta cerca de 69 MB. Hay aproximadamente 11.7x de espacio por explorar con delta encoding lossless denso.

La conclusion correcta es mas especifica:

- Deduplicacion exacta por bloques fijos no sirve para SFT denso.
- Lossless delta encoding denso entre checkpoints consecutivos si parece prometedor.
- El siguiente prototipo debe guardar `zstd(current XOR previous)` o una variante tensorial equivalente, no snapshots completos deduplicados por bloque exacto.

Top de deltas tensoriales por tamano comprimido:

| Tensor | Raw Bytes | `zstd(XOR)` | Ratio |
|---|---:|---:|---:|
| `embed_out.weight` | 103,022,592 | 10,712,444 | 10.40% |
| `gpt_neox.embed_in.weight` | 103,022,592 | 1,350,107 | 1.31% |
| `gpt_neox.layers.1.mlp.dense_4h_to_h.weight` | 8,388,608 | 1,008,573 | 12.02% |
| `gpt_neox.layers.0.mlp.dense_4h_to_h.weight` | 8,388,608 | 915,833 | 10.92% |
| `gpt_neox.layers.7.mlp.dense_4h_to_h.weight` | 8,388,608 | 901,893 | 10.75% |

## Addendum: Breakdown Del Commit

La medicion instrumentada de dos commits consecutivos mostro:

| Checkpoint | Commit Total | Growth Z-Space | Nodos |
|---:|---:|---:|---:|
| 1 | 82.38 s | 810,064,167 | 12,652 |
| 2 | 83.09 s | 800,619,231 | 25,112 acumulados |

Breakdown por checkpoint:

| Fase | Checkpoint 1 | Checkpoint 2 |
|---|---:|---:|
| Construir payload RAW tensorial | 10.14 s | 10.77 s |
| Compresion de scoring del candidato | 20.60 s | 20.91 s |
| Store por bloques | 39.59 s | 39.29 s |
| Calculo de `tree_compressed_size` | 0.65 s | 0.61 s |
| `tensor_digest` | 11.37 s | 11.48 s |

Breakdown interno del store para los dos commits combinados:

| Fase Interna | Tiempo | Bytes |
|---|---:|---:|
| `put_tensor_payload_total` | 78.83 s | 1,621,408,900 |
| Loop de bloques | 77.51 s | 1,620,049,920 |
| Escritura de nodos | 44.58 s | 1,610,683,398 |
| Compresion LZ4 de nodos | 28.19 s | 1,614,249,343 |
| Hash SHA-256 de nodos | 2.96 s | 1,623,727,272 |
| Split payload tensorial | 0.23 s | 1,621,408,900 |
| Construccion de manifiestos | 0.09 s | 2,344,260 |

Conteos:

- Writes de nodos: 25,112
- Bloques tensoriales escritos: 24,577
- Hits de dedup exacto: 192
- Nodos por checkpoint: ~12.5k

No hay `fsync` explicito por nodo en el codigo instrumentado. El cuello es la combinacion de fragmentacion en miles de archivos pequenos, `write_bytes` + `os.replace` por nodo, compresion LZ4 por bloque y trabajo duplicado de serializacion/digest.

La consulta de hardware de Windows (`Get-PhysicalDisk`, `Get-Volume`) fue bloqueada por permisos, pero la prueba secuencial local en `D:` dio:

- Escritura secuencial: 3,123.93 MB/s
- Lectura secuencial: 3,183.55 MB/s

Por tanto, el disco bruto no explica los 82-105 s de commit. El path actual degrada el rendimiento por granularidad de I/O y trabajo CPU redundante.

### Re-medicion Despues De Optimizar `TensorCodec.raw_parts`

Se encontro un error de performance en `TensorCodec.raw_parts`: convertia la vista `uint8` con `tolist()`. En una prueba local de 64 MiB, esa ruta tardaba ~1.33 s; reemplazarla por `byte_view.numpy().tobytes()` bajo a ~0.011 s.

Despues del cambio, el probe de dos checkpoints se repitio:

| Checkpoint | Commit Total | Growth Z-Space | Nodos |
|---:|---:|---:|---:|
| 1 | 67.19 s | 810,064,201 | 12,652 |
| 2 | 67.09 s | 800,618,580 | 25,112 acumulados |

Breakdown por checkpoint despues del fix:

| Fase | Checkpoint 1 | Checkpoint 2 |
|---|---:|---:|
| Construir payload RAW tensorial | 0.34 s | 0.32 s |
| Compresion de scoring del candidato | 20.71 s | 20.34 s |
| Store por bloques | 43.61 s | 44.00 s |
| Calculo de `tree_compressed_size` | 0.69 s | 0.63 s |
| `tensor_digest` | 1.80 s | 1.77 s |

La optimizacion quito ~15 s por commit, pero el commit sigue dominado por:

- compresion redundante de scoring (~20 s);
- escritura/compresion de miles de nodos pequenos (~44 s);
- LZ4 por bloque con ahorro minimo sobre pesos densos.

## Diagnostico

El resultado es negativo para el caso "full SFT dense checkpointing" usando deduplicacion exacta por bloques.

La deduplicacion por bloques exactos no ayuda cuando el entrenamiento actualiza la mayoria de los pesos en cada checkpoint. En SFT denso con AdamW, incluso cambios pequenos alteran los bytes de casi todos los bloques de pesos. Por eso cada snapshot de Pythia-410M se comporta casi como contenido nuevo.

El benchmark no uso `ZSpace.update` con deltas sparse, porque un checkpoint real de SFT no es sparse: cada checkpoint es un `state_dict` completo con cambios densos. El mecanismo probado fue deduplicacion content-addressable entre snapshots completos. Esa ruta falla para SFT denso.

Sin embargo, la medicion `zstd(c_i XOR c_{i+1})` demuestra que un delta lossless denso si puede ser viable.

## Hallazgos

1. El crecimiento por checkpoint con dedup exacta queda alrededor de 800 MB, casi igual al checkpoint PyTorch de 810 MB.
2. El ahorro acumulado observado es solo ~1%, insuficiente para justificar el enfoque.
3. El commit de Z-Space es mucho mas lento que `torch.save`: ~91 s vs ~3.6 s promedio.
4. La fragmentacion en bloques genera muchos nodos: el checkpoint 27 llego a mas de 337k nodos.
5. La deduplicacion por bloques exactos sirve para versionado sparse/sintetico, pero no para fine-tuning denso real.
6. El delta `zstd(XOR)` entre checkpoints consecutivos mide ~69 MB, asi que hay espacio real para una estrategia lossless densa.

## Addendum: Baseline DVC 100 Checkpoints

Se ejecuto un baseline independiente con DVC:

- Nombre formal: "DVC content-addressable storage baseline"
- Script: `scripts/benchmark_dvc_pythia_410m.py`
- DVC: 3.67.1
- Config: Pythia-410M, WikiText, 100 checkpoints, cada 50 steps, bf16, `eval_probe_batches=8`
- Metodo: `torch.save(checkpoint_N.pt)`, `dvc add`, `dvc commit`, borrar el `.pt` del workspace y medir `.dvc/cache`

Resultado:

| Metrica | Valor |
|---|---:|
| Cache final `.dvc/cache` | 81,076,416,700 bytes |
| PyTorch checkpoints acumulados | 81,076,416,700 bytes |
| Ratio DVC / PyTorch | 100.00% |
| Ahorro DVC vs PyTorch | 0.00% |
| Objetos DVC en cache | 100 |
| Growth promedio por checkpoint | 810,764,167 bytes |
| `dvc add + commit` promedio | 12.098 s |
| Pipeline completo promedio | 13.971 s |
| `dvc checkout` final | 1.914 s |

Conclusion: DVC confirma el diagnostico. Para checkpoints bf16 densos, un CAS
byte-exacto guarda practicamente un snapshot completo por version. La ruta que
vale seguir no es dedup exacta, sino delta lossless denso (`XOR+zstd` o
precondicionadores equivalentes).

## Addendum: Safetensors + Zstd Dict 100 Checkpoints

Se ejecuto el baseline independiente "Safetensors + zstd dictionary compression
baseline":

- Script: `scripts/benchmark_safetensors_zstd_baseline.py`
- Config: Pythia-410M, WikiText, 100 checkpoints, cada 50 steps, bf16, `eval_probe_batches=8`
- Serializacion: `safetensors`
- Compresion: `zstd(level=9, threads=-1)`
- Ablacion: sin diccionario, dict 100 KB, dict 1 MB
- Diccionarios entrenados solo con el primer checkpoint, dividido en chunks de 1 MiB
- Payloads comprimidos no escritos a disco; solo se midio `len(compressed)`

Resultado:

| Metrica | Sin dict | Dict 100 KB | Dict 1 MB |
|---|---:|---:|---:|
| Total comprimido | 62,669,100,362 bytes | 62,676,891,806 bytes | 62,667,040,777 bytes |
| Ratio vs `torch.save` | 77.30% | 77.31% | 77.29% |
| Ahorro vs `torch.save` | 22.70% | 22.69% | 22.71% |
| Promedio por checkpoint | 626.691 MB | 626.769 MB | 626.670 MB |
| Compresion promedio | 9.576 s | 9.333 s | 9.499 s |

Conclusion: el baseline industrial honesto mejora mucho sobre CAS byte-exacto,
pero queda muy lejos de Z-Space v2 con delta denso (`17.76%` de PyTorch con
`full_every=8`). El dict no cambia la conclusion: zstd sin dict, 100 KB y 1 MB
son practicamente iguales, lo que indica un techo de compresion per-checkpoint
estable para pesos bf16 densos.

## Causa Raiz

El sistema actual deduplica identidad exacta de bloques. Esa tecnica presupone que partes grandes del tensor permanecen byte-identicas entre versiones.

En SFT denso esa premisa no se cumple. El optimizador toca una fraccion amplia del modelo y propaga diferencias por todo el `state_dict`. Aunque numericamente muchos cambios sean pequenos, byte a byte los bloques dejan de coincidir.

Pero los XOR entre checkpoints si tienen estructura comprimible. La causa raiz del fallo no es "lossless imposible"; es "dedup exacta por bloque equivocado para deltas densos".

## Decision

No continuar la corrida de 100 checkpoints con deduplicacion exacta por bloques en este estado. La extrapolacion desde 27 checkpoints es estable:

- 100 checkpoints PyTorch: ~81.1 GB
- 100 checkpoints Z-Space estimado: ~80.3 GB
- Mejora esperada: marginal
- Tiempo esperado de commits Z-Space: ~2.5 horas solo serializando commits

## Implicacion Tecnica Corregida

Para SFT denso real, el siguiente salto no es mas deduplicacion exacta por bloques. Hacen falta deltas densos eficientes:

- XOR residual entre checkpoints consecutivos con `zstd`, usando la medicion de ~69 MB como baseline inicial.
- Delta tensorial aritmetico (`current - previous`) y comparacion contra XOR para bf16/fp16.
- Agrupar bloques en archivos append-only o packfiles para eliminar miles de writes pequenos.
- Evitar compresion de scoring cuando `decomp_type=RAW` esta forzado.
- Reusar bytes canonicos para `tensor_digest` en vez de serializar dos veces.
- Deltas por threshold o top-k para escenarios donde se pueda tolerar politica sparse.
- Checkpointing de adapters/LoRA en vez de pesos completos.

## Conclusión

Z-Space demostro buen comportamiento en versionado sparse y modelos sinteticos, pero el benchmark real de Pythia-410M invalida la hipotesis de que deduplicacion exacta por bloques sea suficiente para 100 checkpoints de SFT denso.

El sistema necesita una estrategia de delta denso entre checkpoints si el objetivo es ahorrar espacio en entrenamientos reales full fine-tuning. La medicion `zstd(XOR)` indica que esa linea si merece prototipo: ~69 MB por delta frente a ~810 MB por checkpoint completo.

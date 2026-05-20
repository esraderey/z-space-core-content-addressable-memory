# Paper Metrics: Lossless Checkpoint Versioning Benchmarks

Fecha: 2026-05-18

## Resumen Ejecutivo

Este documento resume los resultados empiricos principales para comparar
versionado lossless de checkpoints de entrenamiento denso.

Caso evaluado:

- Modelo: `EleutherAI/pythia-410m`
- Dataset: WikiText `wikitext-103-raw-v1`
- Regimen: SFT real, 5000 steps
- Checkpoints: 100
- Intervalo: cada 50 steps
- Dtype: `bf16`
- Batch size: 1
- Block size: 512
- GPU: NVIDIA GeForce RTX 4060
- Evaluacion de entrenamiento: `eval_probe_batches=8`

Resultado central:

| Metodo | Store total | Ratio vs `torch.save` | Ahorro vs `torch.save` |
|---|---:|---:|---:|
| PyTorch checkpoints | 81.076-81.077 GB | 100.00% | 0.00% |
| DVC CAS baseline | 81.076 GB | 100.00% | 0.00% |
| Safetensors + zstd, sin dict | 62.669 GB | 77.30% | 22.70% |
| Safetensors + zstd, dict 100 KB | 62.677 GB | 77.31% | 22.69% |
| Safetensors + zstd, dict 1 MB | 62.667 GB | 77.29% | 22.71% |
| Z-Space v2, XOR+zstd, sin full periodico | 5.346 GB | 6.59% | 93.41% |
| Z-Space v2, XOR+zstd, `full_every=8` | 14.400 GB | 17.76% | 82.24% |

Conclusion:

- CAS byte-exacto no deduplica checkpoints bf16 densos.
- Compresion per-checkpoint con zstd mejora el baseline bruto, pero se queda en
  ~77.3% del tamano PyTorch.
- Diccionarios zstd entrenados solo sobre el checkpoint base no aportan ahorro
  material: 100 KB y 1 MB quedan practicamente empatados con zstd sin diccionario.
- Delta lossless denso entre checkpoints captura estructura que la compresion
  per-file no ve.
- Sin full periodico, Z-Space v2 logra el mejor ratio medido: 6.59% de
  `torch.save`, a costa de reconstruccion final lenta. Con `full_every=8`,
  sube a 17.76% pero reconstruye el checkpoint final en 18.456 s.

## Metodologia

### Checkpoint Reference

Cada checkpoint corresponde al `state_dict` completo del modelo. Para los
baselines se usa el mismo schedule:

- `checkpoint_1`: step 50
- `checkpoint_2`: step 100
- ...
- `checkpoint_100`: step 5000

El total de referencia es la suma de 100 checkpoints PyTorch:

| Metrica | Valor |
|---|---:|
| Checkpoints | 100 |
| Bytes acumulados `torch.save` | 81,076,601,900 bytes en corrida Z-Space |
| Bytes acumulados `torch.save` | 81,076,416,700 bytes en corrida DVC |
| Bytes acumulados `torch.save` | 81,076,120,700 bytes en corrida safetensors/zstd |
| Tamano por checkpoint | ~810.7 MB |

Para tablas comparativas del paper se usa un denominador comun redondeado de
~81.076 GB. Los bytes exactos por corrida se conservan arriba. Las diferencias
son menores a 0.001% y pueden venir de metadatos de serializacion, variantes
menores de ejecucion y corridas independientes; no cambian ninguna conclusion.

### Zstd Level Policy

Z-Space v2 uso `zstd-level=3` porque el input ya esta pre-acondicionado como
XOR tensorial. El baseline safetensors uso `zstd-level=9` para darle al baseline
per-checkpoint una configuracion fuerte. Para descartar sesgo, se midio una
ablacion corta de 10 checkpoints con Z-Space v2 sin full periodico:

| Zstd level | Store final | Ratio vs PyTorch | Commit promedio | Reconstruccion final |
|---|---:|---:|---:|---:|
| 3 | 1.381 GB | 17.04% | 10.463 s | 60.737 s |
| 9 | 1.334 GB | 16.45% | 17.622 s | 50.089 s |

`zstd-9` mejora solo 0.59 puntos porcentuales en 10 checkpoints, pero aumenta
el commit promedio ~68%. Usar `zstd-3` para Z-Space y `zstd-9` para el baseline
safetensors es conservador hacia los baselines, no favorable a Z-Space.

## Baseline 1: DVC Content-Addressable Storage

Nombre formal:

> DVC content-addressable storage baseline

Politica:

- Guardar `checkpoint_N.pt` con `torch.save`.
- Ejecutar `dvc add checkpoint_N.pt`.
- Ejecutar `dvc commit checkpoint_N.pt.dvc`.
- Borrar el `.pt` del workspace para no duplicar almacenamiento local.
- Medir el tamano acumulado de `.dvc/cache`.

Resultado:

| Metrica | Valor |
|---|---:|
| DVC version | 3.67.1 |
| Checkpoints | 100 |
| Objetos en cache | 100 |
| Cache final `.dvc/cache` | 81,076,416,700 bytes |
| PyTorch acumulado | 81,076,416,700 bytes |
| Ratio DVC / PyTorch | 100.00% |
| Ahorro vs PyTorch | 0.00% |
| Growth promedio por checkpoint | 810,764,167 bytes |
| `torch.save` promedio | 1.874 s |
| `dvc add` promedio | 4.882 s |
| `dvc commit` promedio | 7.216 s |
| `dvc add + commit` promedio | 12.098 s |
| Pipeline completo promedio | 13.971 s |
| `dvc checkout` final | 1.914 s |
| Wall time | 42.52 min |

Interpretacion:

DVC almaceno un objeto completo por checkpoint. El crecimiento fue exactamente
lineal. Esto confirma que CAS byte-exacto no captura estructura util entre
checkpoints bf16 densos de SFT.

## Baseline 2: Safetensors + Zstd Dictionary Compression

Nombre formal:

> Safetensors + zstd dictionary compression baseline

Politica:

- Serializar cada checkpoint independiente con `safetensors`.
- Entrenar diccionarios zstd solo sobre el checkpoint base (`checkpoint_1`).
- Partir el checkpoint base en chunks de 1 MiB.
- Entrenar dos diccionarios:
  - 100 KB
  - 1 MB
- Comparar tres compresores:
  - `zstd(level=9, threads=-1)` sin diccionario
  - `zstd(level=9, threads=-1, dict=100KB)`
  - `zstd(level=9, threads=-1, dict=1MB)`
- No escribir payloads comprimidos; medir `len(compressed)`.

Resultado final:

| Metrica | Sin dict | Dict 100 KB | Dict 1 MB |
|---|---:|---:|---:|
| Total comprimido | 62,669,100,362 bytes | 62,676,891,806 bytes | 62,667,040,777 bytes |
| Total comprimido | 62.669 GB | 62.677 GB | 62.667 GB |
| Ratio vs `torch.save` | 77.30% | 77.31% | 77.29% |
| Ahorro vs `torch.save` | 22.70% | 22.69% | 22.71% |
| Promedio por checkpoint | 626.691 MB | 626.769 MB | 626.670 MB |
| Tiempo promedio de compresion | 9.576 s | 9.333 s | 9.499 s |

Datos del diccionario:

| Metrica | Dict 100 KB | Dict 1 MB |
|---|---:|---:|
| Tamano solicitado | 100,000 bytes | 1,000,000 bytes |
| Tamano real | 100,000 bytes | 1,000,000 bytes |
| Tiempo de entrenamiento | 18.773 s | 18.287 s |
| Samples de entrenamiento | 774 chunks | 774 chunks |
| Bytes usados para entrenamiento | 810,702,160 bytes | 810,702,160 bytes |

Datos adicionales:

| Metrica | Valor |
|---|---:|
| `safetensors` acumulado | 81,070,216,000 bytes |
| `safetensors` vs `torch.save` | 99.99% |
| Serializacion safetensors promedio | 0.576 s |
| Wall time | 78.28 min |
| Payloads comprimidos escritos a disco | No |

Interpretacion:

El dict de 1 MB mejora solo ~0.003 puntos porcentuales contra zstd sin
diccionario. El dict de 100 KB queda levemente peor que zstd sin dict. Por tanto
el techo de compresion per-checkpoint esta alrededor de 77.3% para este caso.
El limite parece venir de la entropia intrinseca de pesos bf16 densos, no del
sub-muestreo del diccionario.

## Z-Space v2: Checkpoint Graph + XOR+Zstd Deltas

Nombre formal:

> Z-Space checkpoint graph with tensor-wise XOR+zstd deltas

Politica:

- Representar checkpoint como `CheckpointDescriptor`.
- Cada checkpoint contiene mapping `key -> tensor descriptor` o `key -> delta descriptor`.
- Cada delta tensorial usa `current XOR parent` comprimido con zstd.
- Reconstruccion exacta via parent checkpoint + delta.
- Politica de maxima compresion: sin full periodico (`full_every=0`).
- Politica operacional: full checkpoint periodico cada 8 versiones
  (`full_every=8`) para limitar el tiempo de reconstruccion.

### Z-Space v2: Sin Full Periodico

Resultado final medido en 100 checkpoints:

| Metrica | Valor |
|---|---:|
| Checkpoints | 100 |
| Steps | 5000 |
| `full_every` | 0 |
| `zstd_level` | 3 |
| Exactitud final | `true` |
| Store final | 5,346,228,743 bytes |
| Store final | 5.346 GB |
| PyTorch acumulado | 81,076,601,900 bytes |
| Ratio Z-Space / PyTorch | 6.59% |
| Ahorro vs PyTorch | 93.41% |
| Reconstruccion final | 533.337 s |
| Wall time | 44.87 min |

Breakdown de growth:

| Metrica | Valor |
|---|---:|
| Base growth | 810,064,829 bytes |
| Deltas | 99 |
| Delta promedio | 45,819,838 bytes |
| Delta mediano | 45,962,606 bytes |
| Delta minimo | 31,959,818 bytes |
| Delta maximo | 68,778,142 bytes |
| Delta commit promedio | 8.639 s |
| Delta commit mediano | 8.429 s |
| Delta promedio `ck90-ck100` | 35,053,855 bytes |

### Z-Space v2: Full Periodico Cada 8

Resultado final medido en 100 checkpoints:

| Metrica | Valor |
|---|---:|
| Checkpoints | 100 |
| Steps | 5000 |
| `full_every` | 8 |
| Exactitud final | `true` |
| Store final | 14,400,116,390 bytes |
| Store final | 14.400 GB |
| PyTorch acumulado | 81,076,601,900 bytes |
| Ratio Z-Space / PyTorch | 17.76% |
| Ahorro vs PyTorch | 82.24% |
| Reconstruccion final | 18.456 s |
| Wall time | 36.61 min |

Breakdown de growth:

| Metrica | Valor |
|---|---:|
| Base growth | 810,064,299 bytes |
| Base commit | 18.743 s |
| Deltas | 87 |
| Delta promedio | 44,509,491 bytes |
| Delta mediano | 42,080,253 bytes |
| Delta minimo | 32,204,132 bytes |
| Delta maximo | 68,517,624 bytes |
| Delta commit promedio | 8.268 s |
| Full checkpoints periodicos | 12 |
| Full growth promedio | 809,810,532 bytes |
| Full commit promedio | 18.765 s |

Checkpoints full periodicos:

```text
9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97
```

Interpretacion:

Con `full_every=8`, Z-Space v2 queda 4.35x mas pequeno que
safetensors+zstd-dict1m y 5.63x mas pequeno que DVC/PyTorch. La mejora proviene
de codificar diferencias densas entre checkpoints, no de deduplicacion
byte-exacta.

La politica sin full periodico reduce el store a 6.59% de PyTorch, pero la
reconstruccion final sube a 533.337 s porque debe aplicar 99 deltas. La politica
`full_every=8` es la configuracion operacional recomendada si la reconstruccion
interactiva importa: 18.456 s de reconstruccion final a cambio de subir el store
a 17.76%.

## Practical Delta Bound: zstd(checkpoint_i XOR checkpoint_i+1)

Medicion de dos checkpoints consecutivos reales:

- `checkpoint_1`: step 50
- `checkpoint_2`: step 100

| Medicion | Base | `zstd(XOR)` | Ratio |
|---|---:|---:|---:|
| Tensorial canonico alineado | 810,668,032 bytes | 67,466,202 bytes | 8.32% |
| Archivo `torch.save` | 810,763,273 bytes | 67,539,689 bytes | 8.33% |

Nota early-vs-late:

Este probe es solo `checkpoint_1 -> checkpoint_2` (`step 50 -> step 100`). No
representa `ck90 -> ck91`, `ck99 -> ck100`, ni el promedio de los 100
checkpoints. En la corrida Z-Space v2 sin full periodico, los deltas tardios
`ck90-ck100` promedian 35.054 MB, casi la mitad del probe temprano de ~67.5 MB.
Por tanto, cualquier uso del 8.32% debe etiquetarse como early-probe bound, no
como bound global de toda la trayectoria.

Interpretacion:

Esto no es un lower bound matematico ni global. Es un baseline practico medido
en `checkpoint_1 -> checkpoint_2`, temprano en el entrenamiento, cuando los
gradientes y los cambios de pesos son mas grandes. En la corrida Z-Space v2 sin
full periodico, los deltas tardios `ck90-ck100` promedian 35.054 MB, contra
~67.5 MB del probe temprano.

Por tanto, que el promedio del sistema completo quede por debajo del probe
`ck1 -> ck2` no contradice el probe: el probe mide el peor tramo temprano, no el
promedio de toda la trayectoria de entrenamiento.

La medicion refuta la conclusion de que "lossless no sirve" para SFT denso. La
conclusion correcta es:

- Deduplicacion exacta por bloques no sirve.
- Compresion per-checkpoint ayuda poco.
- Delta lossless denso si captura estructura entre iteraciones.

## Z-Space v1 / CAS Exacto Por Bloques

Resultado historico parcial, antes de implementar v2:

| Metrica | Valor |
|---|---:|
| Checkpoints registrados | 27 / 100 |
| Store Z-Space | 21,675,513,579 bytes |
| PyTorch equivalente | 21,890,682,513 bytes |
| Ratio Z-Space / PyTorch | 99.02% |
| Growth promedio | 802,796,799 bytes |
| Commit promedio | 91.27 s |
| Commit minimo | 82.60 s |
| Commit maximo | 105.14 s |
| `torch.save` promedio | 3.62 s |

Interpretacion:

Este resultado no debe presentarse como resultado final del sistema. Sirve como
control negativo: demuestra que CAS exacto por bloques fracasa en SFT denso
porque casi ningun bloque permanece byte-identico entre checkpoints.

El salto de ~91 s/commit en v1 a ~8-9 s/delta en v2 no es magico: v1
fragmentaba en bloques pequenos con deduplicacion byte-exacta y generaba muchos
nodos e I/O granular. v2 trabaja a nivel tensor/checkpoint graph, calcula XOR
contra el padre y escribe blobs comprimidos en packfiles append-only. Ese cambio
reduce fragmentacion, indexado y escrituras pequenas, ademas de mejorar el
ratio de almacenamiento.

## Training Sanity Check

Las corridas no estan midiendo ruido de un entrenamiento divergente. El probe de
evaluacion baja durante el entrenamiento.

| Corrida | Eval probe inicial | Eval probe final | Cambio |
|---|---:|---:|---:|
| Z-Space v2 100 checkpoints | 3.3769 | 3.0556 | -9.51% |
| Z-Space v2 sin full periodico | 3.3569 | 3.0545 | -9.01% aprox. |
| DVC baseline 100 checkpoints | 3.3721 | 3.0596 | -9.27% aprox. |
| Safetensors+zstd 100 checkpoints | 3.3675 | 3.0562 | -9.24% aprox. |

## Empirical Observation: Per-Checkpoint Compression Ceiling

La compresion per-checkpoint de checkpoints SFT densos bf16 converge a ~77.3%
del tamano `torch.save`, independientemente de las configuraciones de
diccionario zstd probadas:

| Configuracion | Ratio vs `torch.save` |
|---|---:|
| zstd sin diccionario | 77.30% |
| zstd dict 100 KB entrenado en checkpoint base | 77.31% |
| zstd dict 1 MB entrenado en checkpoint base | 77.29% |

No pretende ser una afirmacion formal. Es una observacion empirica robusta para
esta corrida: sugiere un techo de entropia para compresion independiente por
checkpoint que los metodos cross-version pueden superar.

## Paper-Ready Claims

### Claim 1

Byte-exact content-addressed storage does not deduplicate dense bf16 SFT
checkpoints in practice.

Evidence:

- DVC CAS stores 100 objects for 100 checkpoints.
- Final cache: 81.076 GB.
- Ratio vs PyTorch: 100.00%.

### Claim 2

Per-checkpoint industrial compression improves over raw checkpoints, but remains
far from delta-based versioning.

Evidence:

- Safetensors+zstd level 9: 77.30% of PyTorch.
- Dict 100 KB: 77.31%.
- Dict 1 MB: 77.29%.
- Dictionary size has negligible effect.

### Claim 3

Lossless dense deltas expose substantially more structure than per-checkpoint
compression.

Evidence:

- `zstd(XOR)` between two consecutive checkpoints: ~67.5 MB, 8.32%.
- The 8.32% probe is early-only (`ck1 -> ck2`), not a whole-run lower bound;
  late deltas `ck90-ck100` average 35.1 MB.
- Z-Space v2 no periodic full: 5.346 GB, 6.59% of PyTorch.
- Z-Space v2 `full_every=8` 100-checkpoint run: 14.4 GB, 17.76% of PyTorch.
- Average stored delta in Z-Space v2: 44.5 MB.

### Claim 4

Periodic full checkpoints trade space for reconstruction speed.

Evidence:

- 100-checkpoint run with no periodic full: 6.59% of PyTorch, reconstruction
  533.337 s.
- 12-checkpoint run with `full_every=8`: 23.33% of PyTorch, reconstruction
  19.202 s.
- 100-checkpoint run with `full_every=8`: 17.76% of PyTorch, reconstruction
  18.456 s.

## Comparative Ratios

Against Z-Space v2 operational policy (`full_every=8`):

| Comparison | Ratio |
|---|---:|
| DVC / Z-Space v2 | 5.63x larger |
| Safetensors zstd none / Z-Space v2 | 4.35x larger |
| Safetensors zstd dict100k / Z-Space v2 | 4.35x larger |
| Safetensors zstd dict1m / Z-Space v2 | 4.35x larger |
| PyTorch / Z-Space v2 | 5.63x larger |

Against Z-Space v2 maximum-compression policy (no periodic full):

| Comparison | Ratio |
|---|---:|
| DVC / Z-Space v2 no periodic full | 15.17x larger |
| Safetensors zstd none / Z-Space v2 no periodic full | 11.72x larger |
| Safetensors zstd dict100k / Z-Space v2 no periodic full | 11.72x larger |
| Safetensors zstd dict1m / Z-Space v2 no periodic full | 11.72x larger |
| PyTorch / Z-Space v2 no periodic full | 15.17x larger |

## Reproducibility Artifacts

### Z-Space v2 Full Periodico Cada 8

- Summary: `.zspace_bench/pythia_410m_100_xor_full8_eval8/analysis_summary.json`
- Full results: `.zspace_bench/pythia_410m_100_xor_full8_eval8/results.json`
- CSV: `.zspace_bench/pythia_410m_100_xor_full8_eval8/checkpoints.csv`

Command:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_sft_pythia_410m.py `
  --checkpoints 100 `
  --steps-per-checkpoint 50 `
  --checkpoint-mode xor-zstd `
  --checkpoint-full-every 8 `
  --delta-preconditioner xor `
  --zstd-level 3 `
  --eval-probe-batches 8 `
  --gradient-checkpointing `
  --measure-torch-checkpoints
```

### Z-Space v2 Sin Full Periodico

- Summary: `.zspace_bench/pythia_410m_100_xor_full0_zstd3_20260518/analysis_summary.json`
- Full results: `.zspace_bench/pythia_410m_100_xor_full0_zstd3_20260518/results.json`
- CSV: `.zspace_bench/pythia_410m_100_xor_full0_zstd3_20260518/checkpoints.csv`

Command:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_sft_pythia_410m.py `
  --checkpoints 100 `
  --steps-per-checkpoint 50 `
  --checkpoint-mode xor-zstd `
  --checkpoint-full-every 0 `
  --delta-preconditioner xor `
  --zstd-level 3 `
  --eval-probe-batches 8 `
  --gradient-checkpointing `
  --measure-torch-checkpoints
```

### Z-Space Zstd Level Ablation

- Summary: `.zspace_bench/zspace_zstd_level_ablation_10_20260518/analysis_summary.json`
- Zstd level 3 results: `.zspace_bench/zspace_zstd_level_ablation_10_20260518/zstd3/results.json`
- Zstd level 9 results: `.zspace_bench/zspace_zstd_level_ablation_10_20260518/zstd9/results.json`

### DVC Baseline

- Summary: `.zspace_bench/dvc_baseline_pythia410m_100_20260517_v2/analysis_summary.json`
- Full results: `.zspace_bench/dvc_baseline_pythia410m_100_20260517_v2/results.json`
- CSV: `.zspace_bench/dvc_baseline_pythia410m_100_20260517_v2/checkpoints.csv`

Command:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_dvc_pythia_410m.py `
  --checkpoints 100 `
  --steps-per-checkpoint 50 `
  --eval-probe-batches 8 `
  --gradient-checkpointing
```

### Safetensors + Zstd Baseline

- Summary: `.zspace_bench/safetensors_zstd_ablation_100_20260517_v3/analysis_summary.json`
- Full results: `.zspace_bench/safetensors_zstd_ablation_100_20260517_v3/results.json`
- CSV: `.zspace_bench/safetensors_zstd_ablation_100_20260517_v3/checkpoints.csv`

Command:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_safetensors_zstd_baseline.py `
  --checkpoints 100 `
  --steps-per-checkpoint 50 `
  --eval-probe-batches 8 `
  --gradient-checkpointing
```

## Suggested Figure Captions

### Figure: Storage Growth Across 100 Checkpoints

Storage growth for Pythia-410M SFT checkpoints over 5000 training steps. DVC
content-addressed storage grows linearly with PyTorch checkpoints. Safetensors
with zstd compression reduces storage to ~77.3% but remains per-checkpoint.
Z-Space v2 stores tensor-wise XOR+zstd deltas: the maximum-compression policy
without periodic full checkpoints reaches 6.59% of PyTorch, while the
operational `full_every=8` policy reaches 17.76% with faster reconstruction.

### Figure: Per-Checkpoint Compression Ratios

Per-checkpoint zstd compression ratios remain stable across SFT iterations.
Zstd without dictionary, zstd with a 100 KB base-checkpoint dictionary, and zstd
with a 1 MB base-checkpoint dictionary all converge around 77.3% of PyTorch
checkpoint size, indicating that dictionary size is not the limiting factor.

### Figure: Delta Size Distribution

Z-Space v2 without periodic full checkpoints stores 99 deltas with a mean of
45.8 MB; late deltas `ck90-ck100` average 35.1 MB. The `full_every=8` policy
stores 87 deltas with a mean of 44.5 MB. This explains why the early
`zstd(checkpoint_i XOR checkpoint_i+1)` probe measured ~67.5 MB while the
full-run average is lower.

## Reviewer Notes

Potential reviewer question:

> How can the full run average beat the 8.32% `zstd(XOR)` probe?

Answer:

It does not beat a global bound. The 8.32% probe was measured only on
`ck1 -> ck2`, early in training (`step 50 -> step 100`), when parameter changes
are larger. Later checkpoints are more compressible: in the no-periodic-full
100-checkpoint run, `ck90-ck100` deltas average 35.1 MB. The correct wording is
"early-probe practical bound", not "global lower bound".

Potential reviewer question:

> Why does Z-Space use zstd level 3 while safetensors uses zstd level 9?

Answer:

Z-Space compresses XOR-preconditioned tensor deltas, where level 3 already
captures nearly all ratio benefit. A 10-checkpoint ablation showed zstd-9
improved store ratio from 17.04% to 16.45%, only 0.59 points, while increasing
average commit time from 10.463 s to 17.622 s. Safetensors was measured at
level 9 to give the per-checkpoint baseline a strong configuration.

Potential reviewer question:

> Is zstd dictionary compression a fair baseline?

Answer:

Yes. The dictionary is trained only on the base checkpoint, which is available at
versioning time, and does not use future checkpoints or dataset information.
The ablation includes no dictionary, 100 KB dictionary, and 1 MB dictionary.
All three results are effectively identical, showing that the conclusion is not
an artifact of dictionary size.

Potential reviewer question:

> Is Z-Space v2 just using lossy compression?

Answer:

No. The final checkpoint reconstruction was exact (`final_exact=true`). The
codec is lossless: it stores `current XOR parent` compressed with zstd and
reconstructs tensors by applying XOR against the parent checkpoint.

Potential reviewer question:

> Why not use DVC?

Answer:

DVC is a strong content-addressable baseline for byte-exact file versioning, but
it stores complete checkpoint objects for dense bf16 SFT checkpoints. In this
benchmark, DVC cache size was exactly equal to accumulated PyTorch checkpoint
size.

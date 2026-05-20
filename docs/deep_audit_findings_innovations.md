# Deep Audit: Z-Space Checkpoint Versioning

Fecha: 2026-05-14

## Resumen Ejecutivo

La conclusion anterior era incompleta. El benchmark real de SFT denso no demuestra que lossless este muerto; demuestra que la deduplicacion exacta por bloques fijos es el mecanismo equivocado.

Estado despues de corregir P0:

- `z_space_core.py` ya tiene versionado de checkpoints con deltas densos `XOR+zstd` por tensor (`TensorDeltaCodec`, `ZSpace.register_checkpoint`, `ZSpace.update_checkpoint`, `ZSpace.load_checkpoint_desc`).
- El path `decomp_type=RAW` ya evita el torneo/scoring redundante y almacena directo.
- El benchmark Pythia ya usa `PackfileContentStore` append-only y expone `--checkpoint-mode full|xor-zstd`.

La medicion critica es:

- Checkpoint Pythia-410M step 50: ~810.76 MB
- Checkpoint Pythia-410M step 100: ~810.76 MB
- `zstd(c_100 XOR c_50)` sobre bytes tensoriales canonicos: ~67.47 MB
- Ratio: ~8.32%

Eso deja 11-12x de espacio tecnico por capturar con un delta lossless denso.

El problema actual tiene dos caras:

1. **Modelo de almacenamiento equivocado para SFT denso:** guarda snapshots completos con dedup exacta por bloques, pero casi ningun bloque permanece byte-identico.
2. **Commit path demasiado caro:** incluso para guardar snapshots completos, el pipeline hace trabajo redundante y fragmenta el I/O en miles de archivos.

## Que Esta Pasando

En SFT denso, AdamW modifica la mayoria de los pesos en cada intervalo. Eso destruye la identidad exacta de bloques de 64 KiB. Por eso Z-Space crece ~800 MB por checkpoint, casi igual que PyTorch.

Pero esos cambios no son entropia uniforme. El XOR entre checkpoints consecutivos comprime a ~67 MB con zstd. Los bytes cambian, pero cambian con estructura. Esa es la oportunidad.

## Evidencia Principal

### Benchmark Parcial 27/100

| Metrica | Valor |
|---|---:|
| Checkpoints registrados | 27 |
| Store Z-Space | 21,675,513,579 bytes |
| PyTorch equivalente | 21,890,682,513 bytes |
| Ratio Z-Space / PyTorch | 99.02% |
| Growth promedio | 802,796,799 bytes |
| Commit promedio | 91.27 s |
| `torch.save` promedio | 3.62 s |

### Delta Lossless Consecutivo

| Medicion | Base | `zstd(XOR)` | Ratio |
|---|---:|---:|---:|
| Tensorial canonico | 810,668,032 | 67,466,202 | 8.32% |
| Archivo `torch.save` | 810,763,273 | 67,539,689 | 8.33% |

Nota: esto no es un lower bound matematico; es un baseline practico fuerte. Si una estrategia nueva no se acerca a este orden de magnitud, esta perdiendo informacion explotable.

### DVC CAS Baseline 100/100

Se agrego el baseline formal "DVC content-addressable storage baseline" en
`scripts/benchmark_dvc_pythia_410m.py` y se ejecuto contra Pythia-410M,
WikiText, 100 checkpoints cada 50 steps.

| Metrica | Valor |
|---|---:|
| Checkpoints | 100 |
| Steps | 5000 |
| DVC version | 3.67.1 |
| Cache final `.dvc/cache` | 81,076,416,700 bytes |
| Archivos en cache DVC | 100 |
| PyTorch checkpoints acumulados | 81,076,416,700 bytes |
| Ratio DVC / PyTorch | 100.00% |
| Ahorro DVC vs PyTorch | 0.00% |
| Growth promedio por checkpoint | 810,764,167 bytes |
| `dvc add` promedio | 4.882 s |
| `dvc commit` promedio | 7.216 s |
| `dvc add + commit` promedio | 12.098 s |
| Pipeline `torch.save + dvc add + commit` promedio | 13.971 s |
| `dvc checkout` final | 1.914 s |
| Wall time | 42.52 min |

Evidencia:

- Resultado compacto: `.zspace_bench/dvc_baseline_pythia410m_100_20260517_v2/analysis_summary.json`
- Resultado completo: `.zspace_bench/dvc_baseline_pythia410m_100_20260517_v2/results.json`
- CSV por checkpoint: `.zspace_bench/dvc_baseline_pythia410m_100_20260517_v2/checkpoints.csv`

Interpretacion: DVC guardo un objeto completo por checkpoint; el crecimiento fue
exactamente lineal y no hubo deduplicacion util entre snapshots bf16 densos. Esto
confirma empiricamente que un CAS byte-exacto de archivos/checkpoints completos
no captura la estructura que si aparece en `zstd(XOR)`.

### Safetensors + Zstd Dictionary Baseline 100/100

Se agrego el baseline formal "Safetensors + zstd dictionary compression
baseline" en `scripts/benchmark_safetensors_zstd_baseline.py` y se ejecuto una
ablacion contra Pythia-410M, WikiText, 100 checkpoints cada 50 steps.

Politica:

- Serializar cada checkpoint independiente con `safetensors`.
- Entrenar diccionarios solo sobre el primer checkpoint, en chunks de 1 MiB.
- Comparar `zstd(level=9, threads=-1)` sin diccionario, con dict de 100 KB y con dict de 1 MB.
- No escribir payloads comprimidos; solo medir `len(compressed)` y guardar CSV/JSON.

| Metrica | Sin dict | Dict 100 KB | Dict 1 MB |
|---|---:|---:|---:|
| Total comprimido | 62,669,100,362 bytes | 62,676,891,806 bytes | 62,667,040,777 bytes |
| Ratio vs `torch.save` | 77.30% | 77.31% | 77.29% |
| Ahorro vs `torch.save` | 22.70% | 22.69% | 22.71% |
| Promedio por checkpoint | 626.691 MB | 626.769 MB | 626.670 MB |
| Compresion promedio | 9.576 s | 9.333 s | 9.499 s |

Otros datos:

- `torch.save` acumulado: 81,076,120,700 bytes
- `safetensors` acumulado: 81,070,216,000 bytes
- `safetensors` vs `torch.save`: 99.99%
- Dict 100 KB entrenado en 18.773 s
- Dict 1 MB entrenado en 18.287 s
- Samples de entrenamiento del dict: 774 chunks del checkpoint base
- Wall time: 78.28 min

Evidencia:

- Resultado compacto: `.zspace_bench/safetensors_zstd_ablation_100_20260517_v3/analysis_summary.json`
- Resultado completo: `.zspace_bench/safetensors_zstd_ablation_100_20260517_v3/results.json`
- CSV por checkpoint: `.zspace_bench/safetensors_zstd_ablation_100_20260517_v3/checkpoints.csv`

Interpretacion: la compresion per-checkpoint queda estable alrededor de 77.3%
del `torch.save`. El diccionario no aporta ahorro material; 1 MB mejora solo
~0.003 puntos porcentuales contra zstd sin diccionario. Esto confirma que el
techo del baseline viene de la entropia intrinseca de pesos bf16 densos, no de
sub-muestreo del diccionario.

### Commit Breakdown Post-Fix

Despues de corregir `TensorCodec.raw_parts`, dos commits consecutivos quedaron asi:

| Fase | Checkpoint 1 | Checkpoint 2 |
|---|---:|---:|
| Commit total | 67.19 s | 67.09 s |
| RAW payload | 0.34 s | 0.32 s |
| Candidate scoring compression | 20.71 s | 20.34 s |
| Store por bloques | 43.61 s | 44.00 s |
| `tree_compressed_size` | 0.69 s | 0.63 s |
| `tensor_digest` | 1.80 s | 1.77 s |

El disco no es el cuello bruto:

- Escritura secuencial local: ~3.19 GB/s
- Lectura secuencial local: ~3.07 GB/s

El cuello es el patron de I/O y CPU:

- ~12.5k nodos por checkpoint
- `write_bytes` + `os.replace` por nodo
- LZ4 por bloque sobre pesos que casi no comprimen
- scoring redundante antes de almacenar

## Errores Y Riesgos

### P0: Falta Un Codec De Delta Denso

El sistema no tiene un camino para representar `checkpoint_{i+1}` como delta lossless de `checkpoint_i`. Registra cada tensor como snapshot nuevo. Para SFT denso, eso pierde casi toda la oportunidad medida por `zstd(XOR)`.

Impacto: ~800 MB por checkpoint en vez de ~67 MB potenciales.

### P0: Scoring Redundante En RAW Forzado

Cuando `decomp_type=RAW`, el sistema aun comprime el payload RAW completo para medir `candidate.compressed_size()`. En el probe esto costo ~20 s por checkpoint.

Ubicacion:

- `CodecTournament.choose()` calcula `candidate.compressed_size()`.
- `_Candidate.compressed_size()` llama `ReversibleCompressor.compressed_size(payload)`.

Esto es evitable cuando el usuario fuerza `RAW`: no hay torneo real que decidir.

### P0: Store Fragmentado En Miles De Archivos Pequenos

Cada checkpoint Pythia-410M escribe ~12.5k nodos. Aunque el disco secuencial da ~3 GB/s, el path real escribe ~800 MB en ~44 s solo dentro del store.

Impacto: commit lento estructural, no por falta de SSD.

### P1: LZ4 En Bloques De Pesos Densos Ahorra Poco Y Cuesta Mucho

En dos commits:

- Bytes procesados por `node_compress`: ~1.61 GB
- Tiempo `node_compress`: ~28 s
- Store final: ~1.61 GB

Para pesos bf16 densos, comprimir cada bloque RAW tiene poco retorno. La compresion deberia ser adaptativa: si una muestra del bloque no mejora suficiente, guardar raw.

### P1: `TensorCodec.raw_parts` Era Un Cuello Grave

Antes usaba `byte_view.tolist()`. Micro-medicion:

- 64 MiB: ~1.33 s
- Ruta `numpy().tobytes()`: ~0.011 s

Ya corregido en `z_space_core.py`.

Impacto post-fix: commit bajo de ~82-83 s a ~67 s en el probe de dos checkpoints.

### P1: Benchmark `--clear-store` Podia Borrar Su Propio Directorio De Logs

El benchmark usaba por defecto `.zspace_bench/pythia_410m_wikitext` como store y tambien como directorio de resultados. En Windows, al redirigir logs ahi, `--clear-store` choco contra archivos abiertos.

Ya corregido: el default del store ahora es `.zspace_bench/pythia_410m_wikitext/store`.

### P1: `DiskContentStore` No Es Reanudable

El store en disco mantiene `_kinds` y `_sizes` solo en memoria. Si el proceso se reinicia, los blobs siguen en disco pero el indice no existe. Sirve para benchmark de una corrida, no para producto.

### P2: `compressed_bytes` En Descriptor No Es Growth Incremental

`tree_compressed_size()` calcula el tamano del arbol referenciado, no los bytes nuevos agregados al store. Si un bloque ya existia por otro tensor/checkpoint, el descriptor igual lo cuenta. Para auditorias de growth, hay que usar `store.stats()` antes/despues.

### P2: Delta Chains Son Listas Re-serializadas

Cada update serializa la lista completa de deltas. Para muchos updates sobre el mismo tensor esto escala mal. Mejor: delta nodes enlazados (`delta_node` con parent delta).

### P2: Uso De APIs Privadas En Scripts

Los stores de benchmark importan `_digest`, `_TENSOR_BLOCK_MANIFEST_FORMAT`, `_stable_json_bytes`, etc. desde `z_space_core.py`. Eso es fragil. Si el formato se vuelve oficial, debe exponerse como API interna estable o moverse al core.

## Innovaciones Propuestas

### 1. `XorZstdCheckpointDelta`

Nuevo modo de versionado para checkpoints completos:

- Parent: descriptor de checkpoint anterior.
- Para cada tensor con misma forma/dtype: `xor = raw(current) XOR raw(parent)`.
- Comprimir `xor` con zstd.
- Guardar manifiesto por tensor: key, dtype, shape, raw_nbytes, codec, compressed_delta_node.
- Reconstruccion: cargar parent, descomprimir delta, XOR y materializar tensor.

Baseline esperado por medicion real:

- Pythia-410M: ~67 MB por checkpoint delta.
- Contra checkpoint completo: ~12x menos espacio.

Riesgo principal:

- Reconstruccion final desde 100 deltas puede ser lenta si se aplica cadena completa. Solucion: checkpoints base cada N versiones o skip-deltas.

### 2. Delta Streaming Sin Materializar Todo

Estado: implementado en `TensorDeltaCodec`.

El probe actual hace `bytes(a ^ b for ...)`, que es lento y materializa buffers grandes. Implementacion correcta:

- Procesar por chunks grandes.
- Usar `numpy.bitwise_xor` sobre vistas `uint8` o `torch.bitwise_xor`.
- Enviar chunks a `zstd.ZstdCompressor().compressobj()` o stream writer.

La implementacion actual usa chunks de 16 MiB por defecto, `numpy.bitwise_xor`
cuando esta disponible, `ZstdCompressor().compressobj()` para escribir el
stream comprimido y `ZstdDecompressor().stream_reader()` para reconstruir sin
materializar el XOR completo.

Objetivo:

- Commit delta <10 s por Pythia-410M en RTX 4060/SSD local.

### 3. Packfile Store

Sustituir miles de archivos por segmentos append-only:

- `segment-000001.zspack`
- indice: digest -> segment, offset, packed_len, raw_len, kind
- escribir muchos nodos en una sola secuencia;
- fsync por segmento/checkpoint, no por nodo;
- compaction opcional.

Esto ataca directamente los ~44 s de store por checkpoint.

### 4. Adaptive Compression

Politica:

- Para blobs RAW de pesos: probar una muestra pequena.
- Si ahorro <2-5%, guardar sin compresion.
- Para XOR deltas: usar zstd, porque la medicion demuestra estructura.

Esto elimina gran parte de los ~28 s de LZ4 por dos commits.

### 5. Fast Path Para RAW Forzado

Si `decomp_type=RAW`:

- no construir candidatos extra;
- no hacer `ReversibleCompressor.compressed_size()` para scoring;
- almacenar directo;
- calcular digest desde el payload ya construido.

Esto recupera ~20 s por checkpoint en el benchmark.

### 6. Canonical Tensor Bytes Reusables

Crear un objeto interno:

```python
TensorBytes(info, raw, payload, digest)
```

Y usarlo para:

- store;
- tensor digest;
- XOR delta;
- metricas.

Evita serializar el mismo tensor varias veces.

### 7. Version Graph De Checkpoints, No Solo Tensores

Estado: implementado en `ZSpace`.

Hoy `ZSpace` versiona tensores individuales. Para checkpoints reales hace falta:

- `CheckpointDescriptor`
- mapping key -> tensor descriptor/delta descriptor
- parent checkpoint address
- policy de full checkpoint cada N deltas

Asi el benchmark representa el objeto real que se quiere versionar.

La implementacion actual agrega:

- `checkpoint_history(name)` para recorrer el grafo desde la base hasta la version actual.
- `full_every` en `update_checkpoint(...)` para materializar un checkpoint completo cada N versiones.
- `requires_parent=false` en checkpoints full periodicos: conservan `parent_addr` para lineage, pero no cargan el padre para reconstruccion.
- Flags de benchmark: `--checkpoint-full-every` y `--delta-preconditioner`.

### 8. Zstd Dictionary O Delta-Aware Preconditioners

Estado: groundwork implementado; falta benchmark comparativo.

Despues de `XorZstd` baseline:

- probar zstd dictionaries por familia de tensor;
- comparar XOR vs resta aritmetica para bf16/fp16;
- separar exponent/mantissa bytes;
- ordenar bytes por bitplane antes de zstd.

Esto puede bajar de ~67 MB si la estructura numerica es aprovechable.

La implementacion actual agrega al codec:

- `preconditioner="xor"` como default compatible.
- `preconditioner="u16-sub"` para resta modular lossless sobre palabras de 16 bits, util para fp16/bf16.
- `zstd_dict` opcional por tensor, con digest verificado en reconstruccion.

Pendiente: entrenar dictionaries por familia real de tensor y comparar ratios contra XOR en checkpoints consecutivos reales.

## Plan Tecnico Recomendado

1. Implementar `TensorDeltaCodec.XOR_ZSTD`.
2. Implementar checkpoint descriptor con parent y reconstruccion.
3. Hacer benchmark step 50 -> 100 -> 150 con reconstruccion exacta.
4. Medir:
   - growth por delta;
   - commit delta;
   - reconstruccion final;
   - ratio vs `torch.save`;
   - memoria pico.
5. Despues optimizar store con packfiles.

## Criterios De Exito

Para Pythia-410M WikiText:

- Growth delta: <=100 MB por checkpoint.
- Commit delta: <=15 s inicialmente; objetivo <=5-10 s.
- Reconstruccion checkpoint final con 10 deltas: exacta y <=60 s.
- Store total para 100 checkpoints: claramente menor que 20 GB, objetivo inicial ~7-12 GB mas bases periodicas.

## Conclusion

La innovacion no es seguir refinando dedup exacta por bloques. La innovacion correcta es convertir Z-Space en un sistema de checkpoint deltas densos, con `XOR+zstd` como baseline demostrado y packfiles para que el commit sea I/O eficiente.

La auditoria cambio el diagnostico:

- Antes: "SFT denso mata lossless".
- Ahora: "SFT denso mata dedup exacta; lossless delta denso tiene margen real".

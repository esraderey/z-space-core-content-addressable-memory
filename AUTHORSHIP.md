# Carta de Autoría — Z-Space Core

**Proyecto:** Z-Space Core — Content-Addressable Tensor Memory
**Autor:** Raul Cruz Acosta
**Fecha de declaración:** 20 de mayo de 2026
**Repositorio canónico:** `z-space-core-content-addressable-memory`

---

## 1. Declaración de autoría

Yo, **Raul Cruz Acosta**, declaro de forma expresa, voluntaria y bajo mi propia
responsabilidad que soy el **único autor y titular originario** de la obra
denominada **Z-Space Core**, incluyendo, sin limitación:

- el código fuente contenido en este repositorio, en particular el módulo
  [`z_space_core.py`](z_space_core.py);
- los scripts de smoke testing, auditoría y benchmark en `scripts/`;
- las pruebas en `tests/`;
- la documentación técnica en `docs/`;
- el diseño arquitectónico, la nomenclatura interna y las decisiones de API
  pública;
- los algoritmos y técnicas descritos en la Sección 3 de este documento.

Esta obra es resultado de mi trabajo personal, intelectual y técnico. No deriva
de ningún encargo laboral, contrato de obra, beca, convenio universitario ni
acuerdo previo de cesión de derechos con tercero alguno. Ninguna persona física
o jurídica distinta del firmante posee derechos patrimoniales o morales sobre
esta obra a la fecha de esta declaración.

---

## 2. Naturaleza y fecha de la creación

El desarrollo de Z-Space Core se inició en **2026** y continúa en evolución
activa. Las marcas temporales (`timestamps`) del sistema de control de versiones
Git asociadas a este repositorio constituyen evidencia objetiva del proceso
incremental de creación, así como del momento en que cada contribución técnica
fue introducida por el autor.

La presente publicación pública del repositorio bajo la licencia restrictiva
adjunta ([`LICENSE`](LICENSE)) tiene además el efecto de **divulgación
fechada**: a partir de la fecha de cada commit, el contenido correspondiente
queda en el dominio del estado de la técnica con autoría atribuida a Raul Cruz
Acosta.

---

## 3. Contribuciones técnicas reivindicadas

El autor reivindica como contribuciones originales, en el sentido de la
propiedad intelectual y, en su caso, de la propiedad industrial, las siguientes
aportaciones materializadas en el código de este repositorio:

1. **Esquema canónico de direccionamiento por contenido mediante SHA-256**
   aplicado a descriptores y nodos tensoriales, con espacios de nombre
   (`namespaces`) separados para descriptor, nodo y delta.

2. **Almacén tensorial tipo Merkle-DAG**, en el cual los descriptores apuntan a
   nodos por su hash de contenido, permitiendo la deduplicación automática de
   sub-tensores idénticos entre versiones, checkpoints o réplicas de un modelo.

3. **Codec tournament** — selección automática del codec más eficiente entre
   `RAW`, `SPARSE`, `SVD` y estrategias opcionales basadas en TensorLy, en
   función de la fidelidad objetivo y la ratio de compresión observada.

4. **Reconstrucción híbrida lossy + exacta mediante residuo XOR bit a bit**
   sobre una aproximación de bajo rango, permitiendo cargas progresivas
   (`exact=False` para aproximación rápida, `exact=True` para reconstrucción
   verificada bit a bit).

5. **Cadena reversible de deltas** (`add`, `mul`, `patch`) con captura de los
   valores anteriores, permitiendo la versionado de modelos con crecimiento
   proporcional al delta y no al checkpoint completo, así como el `revert`
   determinista.

6. **Serialización segura sin `pickle`** de componentes y deltas a través de
   `mscs`, eliminando la superficie de ataque clásica de los formatos de
   serialización arbitraria de Python.

7. **Disciplina de caché por clonación** — la caché LRU devuelve siempre clones
   del tensor reconstruido, preservando la inmutabilidad del contenido
   direccionado.

Las anteriores contribuciones, en su combinación específica dentro de un único
sistema de memoria tensorial direccionable por contenido con síntesis
determinista, constituyen el núcleo inventivo de Z-Space Core.

---

## 4. Reserva de derechos

El autor se reserva, de forma íntegra y sin renuncia tácita ni expresa, la
totalidad de los derechos patrimoniales y morales sobre la obra, en particular:

- el derecho de reproducción, distribución, comunicación pública y
  transformación;
- el derecho de paternidad e integridad de la obra;
- el derecho a decidir el régimen de licenciamiento futuro (académico,
  comercial, abierto o mixto);
- el derecho a iniciar, en su caso, procedimientos de protección por patente,
  modelo de utilidad o secreto industrial sobre los componentes que sean
  susceptibles de tal protección bajo la legislación aplicable.

Hasta que se publique un régimen de licenciamiento sucesor, **la licencia
vigente y aplicable es exclusivamente la contenida en el fichero [`LICENSE`](LICENSE)**
de este repositorio.

---

## 5. Prohibición expresa de uso para entrenamiento de IA

El autor **prohíbe expresamente** el uso de esta obra, en todo o en parte, como
material de entrenamiento, ajuste fino (`fine-tuning`), evaluación o contexto
de cualquier sistema de inteligencia artificial, modelo de lenguaje, motor
generativo o pipeline de aprendizaje automático, sin autorización previa,
expresa y por escrito del autor.

Esta prohibición se entiende sin perjuicio de los derechos generales reservados
en la Sección 4 y en [`LICENSE`](LICENSE).

---

## 6. Buena fe y veracidad

El autor declara, bajo su responsabilidad, que la información contenida en esta
carta de autoría es veraz y que no existen, a su conocimiento, derechos de
terceros, obligaciones contractuales o circunstancias previas que contradigan
las afirmaciones aquí realizadas. Cualquier inexactitud sobrevenida será
corregida mediante actualización de este documento.

---

## 7. Canales de contacto y solicitudes de licencia

Para cualquiera de los siguientes supuestos:

- solicitudes de licencia académica;
- solicitudes de licencia comercial;
- propuestas de colaboración o de cesión limitada de derechos;
- notificaciones de uso indebido o de presunta infracción;
- consultas sobre atribución o citación académica;

la comunicación deberá dirigirse al autor a través de los canales identificados
en el perfil público del repositorio. La ausencia de respuesta del autor a
cualquier solicitud **no** podrá interpretarse como consentimiento tácito.

---

## 8. Firma

Suscrito y declarado por:

**Raul Cruz Acosta**
Autor y titular originario de Z-Space Core
20 de mayo de 2026

---

*Este documento, en conjunto con el fichero [`LICENSE`](LICENSE) y el historial
de commits del repositorio, constituye la declaración formal de autoría y
titularidad de Z-Space Core.*

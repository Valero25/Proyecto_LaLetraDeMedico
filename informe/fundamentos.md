# Fundamentos: de la neurona artificial a los modelos del proyecto

Esta sección relaciona los principios de redes neuronales vistos en el curso (perceptrón, funciones de activación, función de pérdida, descenso del gradiente y retropropagación) con los componentes que el equipo entrenó en LetraDeMédico. Cada principio se ilustra con una decisión o un resultado medido del proyecto.

## 1. Asociación y aprendizaje por contraste (RAG, Fase 2)

El asociacionismo describe el aprendizaje como la formación de vínculos entre ideas. Aristóteles propuso cuatro leyes de asociación: contigüidad, frecuencia, similitud y contraste. El conexionismo de Bain (1873) ubicó ese conocimiento en las **conexiones** entre elementos, y Hebb (1949) lo llevó a las neuronas: las conexiones entre unidades que se activan juntas se refuerzan.

El ajuste contrastivo del modelo de embeddings (`training/rag/ajustar_embeddings.py`) aplica las cuatro leyes:

- **Contigüidad:** cada par de entrenamiento une una consulta con el fragmento del que se generó. Aparecen juntos en el mismo ejemplo y por eso quedan vinculados.
- **Frecuencia:** cada fragmento tiene hasta 6 consultas distintas y el modelo las ve en 3 épocas. Cuanto más se repite el vínculo, más se fortalece.
- **Similitud:** cada consulta se acerca a su fragmento correcto de MedlinePlus en el espacio vectorial (ej. "tengo el LDL por las nubes" con la página de colesterol).
- **Contraste:** cada consulta se aleja de un **negativo difícil**, elegido entre las confusiones medidas (HbA1c frente a hemoglobina, ALT frente a AST, sangre frente a orina), y de los demás fragmentos del lote.

El material del curso describe la regla de Hebb pura como fundamentalmente inestable, por tres razones. El ajuste contrastivo resuelve dos de ellas de forma directa:

| Problema de la regla de Hebb | Cómo se trata en el proyecto |
|---|---|
| Sin noción de competencia | La pérdida (`MultipleNegativesRankingLoss`) es una softmax sobre las similitudes de la consulta con **todos** los fragmentos del lote. Lo que cuenta es la similitud con el positivo **en relación** con la de los negativos. Si todas suben por igual, la pérdida no cambia: un fragmento solo gana si le gana a los demás. |
| El aprendizaje no tiene límites | La similitud es el coseno entre vectores, que siempre está entre −1 y 1, multiplicado por una escala fija de 20. Ninguna asociación puede crecer sin límite. Además, la pérdida deja de bajar cuando el positivo ya supera con holgura a los negativos. |
| Sin reducción de pesos | **No se aplicó en el ajuste del RAG**: `SentenceTransformerTrainingArguments` usa por defecto un decaimiento de pesos (*weight decay*) de 0, y el script no lo cambió. Sí se aplica en los clasificadores de guardrails (0,01 en AdamW, sección 6), donde encoge en cada paso los pesos que no se necesitan. En el RAG, lo que evita que los pesos se alejen demasiado es la tasa de aprendizaje baja con pocas épocas (sección 8). Agregarlo es un experimento pendiente. |

**Resultado:** en los 12 analitos que el modelo nunca vio, ALIA ajustado sube de 84,4 % a 91,2 % de acierto en el primer resultado (hit@1), y e5-small de 88,8 % a 96,2 %.

## 2. La neurona de McCulloch y Pitts y el filtro de palabras

El modelo de McCulloch y Pitts (1943) es una unidad de umbral con pesos fijos: suma entradas binarias y se activa si se supera un umbral, pero **no tiene mecanismo de aprendizaje**. Los pesos los pone quien la diseña.

El filtro de guardrails original de la app es exactamente esa unidad: recibe como entradas la presencia de 4 palabras ("tienes", "diagnóstico", etc.), cada una con peso 1, y se activa si aparece al menos una (un OR lógico). Como nada se aprende de los datos, el filtro no se adapta al lenguaje real:

- Deja pasar el **85 %** de las frases riesgosas (276 de 323 en test), como "Esto es una anemia".
- Marca 60 frases informativas: 30 por usar "tienes" ("tienes el valor dentro del rango") y 30 por usar "diagnóstico".
- F1 binario: **0,219**.

El modelo de McCulloch y Pitts distingue además dos tipos de sinapsis. Las **excitadoras** transmiten la entrada ponderada. Las **inhibidoras** impiden la activación de la neurona, sin importar las demás entradas. El filtro solo tiene entradas excitadoras. Por eso no puede reconocer una negación ni un descargo de responsabilidad. De hecho, 30 de sus 60 falsos positivos son el propio aviso de seguridad que el asistente debe dar ("Solo tu médico puede hacer un diagnóstico preciso..."). Una entrada inhibidora para "solo tu médico" corregiría esos casos. Pero sería otra regla puesta a mano, y el problema volvería con la siguiente frase.

Este resultado es la motivación de toda la Fase 3: sustituir pesos puestos a mano por pesos aprendidos.

## 3. El perceptrón y la separabilidad lineal (baseline TF-IDF)

El perceptrón de Rosenblatt calcula una suma ponderada de las entradas más un sesgo, $z = \mathbf{w}^\top \mathbf{x} + b$, y le aplica una función de activación. El sesgo reemplaza al umbral ($b = -t$) y se puede tratar como un peso más, conectado a una entrada fija igual a 1. Su teorema de convergencia garantiza que el algoritmo encuentra una frontera correcta **solo si las clases son linealmente separables**. Si no lo son, los pesos no dejan de cambiar. El ejemplo clásico es el XOR, que necesita al menos una capa oculta.

El baseline clásico de la Fase 3 (TF-IDF de 1 y 2 palabras + regresión logística, en `evaluacion/evaluar_guardrails.py`) es un perceptrón de una sola capa con activación sigmoide, generalizada a 3 clases:

- Las entradas $\mathbf{x}$ son los pesos TF-IDF de **2.723** palabras y pares de palabras. Solo se cuentan las que aparecen en al menos 2 frases de entrenamiento.
- Hay un vector de pesos y un sesgo por clase: 2.723 × 3 + 3 = **8.172 parámetros**. La activación es una softmax, la generalización de la sigmoide a 3 clases.
- Los pesos se ajustan minimizando la entropía cruzada más una penalización L2 sobre los pesos (C = 10), el mismo principio que el decaimiento de pesos de la sección 6. El optimizador es de segundo orden (L-BFGS), en lugar del descenso del gradiente simple.
- A diferencia de las redes con capas ocultas, esta función de pérdida es **convexa**: tiene un único mínimo global y el resultado no depende de la inicialización (ver sección 8).

**Lectura de los resultados a la luz del perceptrón:**

| Tarea | F1 de TF-IDF (lineal) | F1 de MrBERT-biomed (no lineal, preentrenado) |
|---|---|---|
| Binaria: ¿frase riesgosa? | 0,977 | 0,991 |
| 3 clases (F1 macro) | 0,929 | 0,973 |
| Clase diagnóstica | 0,890 | 0,961 |

La tarea binaria es **casi linealmente separable** en el espacio de palabras: un clasificador lineal ya acierta en casi todo. Lo que no se separa con una línea son los casos sutiles, que dependen del sentido de la frase y no de sus palabras sueltas: distinguir una frase diagnóstica de una alarmista, o reconocer como diagnóstica una frase que descarta una enfermedad con certeza ("como tu potasio está normal, no hay riesgo de arritmias"). Ahí es donde un modelo no lineal aporta más: 7 puntos en la clase diagnóstica.

## 4. Funciones de activación

Una función de activación decide cuándo se activa una neurona. Si fuera lineal, cualquier red, sin importar cuántas capas tenga, colapsaría en un clasificador lineal con la misma capacidad limitada de la sección anterior. Por eso las redes del proyecto usan activaciones no lineales, cada una elegida por una razón:

| Dónde | Activación | Por qué |
|---|---|---|
| Compuertas de la LSTM (red propia) | Sigmoide | Su rango de 0 a 1 funciona como una compuerta: cuánto olvidar, cuánto escribir, cuánto leer de la memoria. Es continuamente diferenciable, pero no es simétrica en el origen. |
| Estado de la LSTM | Tangente hiperbólica | Es una sigmoide escalada y simétrica en el origen (rango de −1 a 1). |
| Cabeza de clasificación de la red propia | ReLU | Es lineal por tramos y fácil de calcular. Da representaciones dispersas y no satura para valores positivos, así que evita el desvanecimiento del gradiente. Su riesgo son las **neuronas muertas**: si una neurona queda siempre en la zona negativa, su gradiente es cero y deja de aprender. |
| Capas internas de MrBERT (ModernBERT) | GeGLU, una variante con GELU | GELU es una versión suave de ReLU. En GeGLU, la capa proyecta a 2 × 1.152 dimensiones: una mitad pasa por GELU y multiplica a la otra, que actúa como compuerta, como en la LSTM. |
| Cabeza de clasificación que se agregó a MrBERT | SiLU | `Linear(768→768) → SiLU → LayerNorm → Linear(768→3)`. SiLU ($x \cdot \sigma(x)$) también es una versión suave de ReLU. Como es distinta de cero para valores negativos pequeños, reduce el problema de las neuronas muertas. Es la única parte de MrBERT que empieza con pesos aleatorios (592.899 parámetros). |
| Salida de todos los clasificadores | Softmax | Convierte las 3 puntuaciones en probabilidades de informativa, diagnóstica y alarmista. |

La sigmoide y la tangente hiperbólica tienen derivadas cercanas a cero cuando se saturan, lo que produce el **desvanecimiento del gradiente** en secuencias largas. La LSTM existe para mitigar ese problema: su memoria se actualiza sumando, no multiplicando, y el gradiente sobrevive a lo largo de la frase.

## 5. De perceptrones a redes: arquitectura y conteo de parámetros

Una red neuronal es una colección de perceptrones conectados, y sus parámetros aprendibles son los pesos (las aristas) más los sesgos de cada neurona. El mismo ejercicio del curso aplicado a la red propia (`training/guardrails/red_propia.py`) da su tamaño exacto:

| Capa | Cálculo | Parámetros |
|---|---|---|
| Embedding (vocabulario de 988 palabras, dimensión 128) | 988 × 128 | 126.464 |
| BiLSTM, por dirección: 4 compuertas con pesos de entrada, pesos recurrentes y 2 sesgos | 4 × (128×128 + 128×128 + 128 + 128) = 132.096 | |
| BiLSTM, 2 direcciones | 2 × 132.096 | 264.192 |
| Linear 512 → 128 (la entrada es la media y el máximo de las dos direcciones: 4 × 128) | 512 × 128 + 128 | 65.664 |
| Linear 128 → 3 | 128 × 3 + 3 | 387 |
| **Total** | | **456.707** |

El 58 % de los parámetros está en la BiLSTM y el 28 % en los embeddings, que la red debe aprender desde cero con solo 899 frases.

La cabeza de clasificación (las dos capas `Linear`) es exactamente la red retroalimentada (*feed-forward*) del curso. Tiene 512 entradas, una capa oculta de 128 neuronas con ReLU y una capa de salida de 3 neuronas. Son 131 neuronas, 65.920 pesos (512 × 128 + 128 × 3) y 131 sesgos. La BiLSTM, en cambio, es una red **recurrente**: cada paso recibe la palabra actual y además su propio estado del paso anterior. Por eso los pesos recurrentes (128 × 128 por compuerta) se reutilizan en cada palabra de la frase. La retropropagación recorre esa cadena hacia atrás, palabra por palabra; se la llama retropropagación a través del tiempo.

## 6. Poder de representación y sobreajuste

El teorema de aproximación universal (Cybenko, 1989) dice que una red con una sola capa oculta puede aproximar cualquier función continua, y que el poder de representación crece con más capas y neuronas. El material del curso añade la advertencia: un gran poder conlleva un gran sobreajuste.

La red propia lo muestra con claridad. Con unos 500 parámetros por frase de entrenamiento, memoriza el conjunto de entrenamiento mucho antes de generalizar (semilla 13):

| Época | Pérdida en train | F1 macro en validación |
|---|---|---|
| 1 | 1,006 | 0,570 |
| 4 | 0,180 | 0,889 |
| 12 | 0,020 | **0,944** (mejor época) |
| 17 | 0,007 | 0,930 (parada temprana) |

Desde la época 4 la pérdida de entrenamiento sigue cayendo casi hasta cero, mientras que el F1 en validación oscila entre 0,86 y 0,94. En test, sobre analitos no vistos, baja a 0,902 ± 0,009 (3 semillas). Para contenerlo se usaron cuatro mecanismos:

1. **Dropout** de 0,3 en los embeddings y en la cabeza.
2. **Decaimiento de pesos** (weight decay de 0,01 en AdamW), que penaliza los pesos grandes.
3. **Parada temprana:** se guarda la mejor época por F1 en validación y se detiene tras 5 épocas sin mejora.
4. **Split por analito:** validar y evaluar sobre exámenes que el modelo no vio, para que la cifra reportada mida generalización y no memoria.

El contraste con los transformers indica que el número de parámetros por sí solo no explica el rendimiento. MrBERT tiene 308 millones de parámetros, 675 veces más que la red propia, y también memoriza el entrenamiento: su pérdida en train llega a **0,0009**, más baja que la de la red propia (0,007). La diferencia está en la generalización. Con ese mismo nivel de memorización, su F1 en validación se mantiene en 0,99 y en test llega a 0,973. Esos parámetros ya llegan entrenados con millones de textos en español, y el ajuste fino solo los mueve un poco.

La red propia, en cambio, queda **por debajo** del perceptrón lineal de TF-IDF (0,902 frente a 0,929): con tan pocos datos, más capacidad sin conocimiento previo es más riesgo de sobreajuste, no más acierto. Es la fórmula del curso para explicar el aprendizaje profundo: *algoritmos conocidos + potencia de cómputo + grandes datos*. La red propia tiene el algoritmo, pero no los datos. MrBERT recibe los datos a través de su preentrenamiento.

## 7. La función de pérdida: el equipo la elige

La función de pérdida define qué significa estar cerca de la solución correcta, y no hay una única opción válida: depende de la tarea. El proyecto usa dos:

- **Entropía cruzada** en los tres clasificadores de guardrails (BiLSTM, MrBERT-biomed y MrBERT-es). El curso presenta el error cuadrático medio (MSE) como la pérdida típica de la regresión. Para clasificación con softmax se prefiere la entropía cruzada, porque su gradiente respecto a las puntuaciones es simplemente $\hat{p} - y$ (probabilidad predicha menos etiqueta real). Con MSE, ese gradiente se multiplica por la derivada de la softmax, que tiende a cero cuando el modelo se equivoca con mucha seguridad. Justo cuando más hay que corregir, aprendería más lento.
- **Pérdida contrastiva de ranking** (`MultipleNegativesRankingLoss`) en el RAG. Es también una entropía cruzada, pero sobre las similitudes coseno entre una consulta y todos los fragmentos del lote: la "clase correcta" es su fragmento positivo y las demás son los negativos.

**Pérdida de entrenamiento frente a métrica de evaluación.** El curso presenta también la pérdida cero-uno, que vale 1 si la predicción es incorrecta y 0 si es correcta. La exactitud y el F1 con que se reportan los resultados se basan en ella: cuentan aciertos. Pero la pérdida cero-uno es escalonada, y su derivada es cero en casi todas partes, así que no le dice al descenso del gradiente hacia dónde moverse. Por eso **se entrena con entropía cruzada**, una aproximación suave y diferenciable, y **se evalúa con F1**. La entropía cruzada sigue bajando aunque las predicciones ya sean correctas, porque premia la confianza. Por eso la pérdida en train de la red propia llega a 0,007 mientras su F1 en validación ya no mejora (sección 6).

### 7.1 Experimento: pérdida ponderada frente a umbral

La entropía cruzada estándar penaliza igual todos los errores. En un guardrail, en cambio, un falso negativo (dejar pasar una frase diagnóstica al paciente) es más grave que un falso positivo (reescribir una frase que estaba bien). Se probaron dos formas de inclinar la balanza hacia la sensibilidad:

1. **Pérdida ponderada por clase:** $L = -w_y \log \hat{p}_y$, con $w = (1, 2, 2)$ para (informativa, diagnóstica, alarmista). Equivocarse con una frase riesgosa cuesta el doble. Los pesos se fijaron de antemano, sin mirar el test.
2. **Umbral de decisión:** marcar la frase si $P(\text{diagnóstica}) + P(\text{alarmista}) \geq \tau$, con $\tau < 0{,}5$. No cambia la pérdida ni requiere reentrenar.

| Modelo | FN / FP sin ponderar (τ = 0,5) | FN / FP ponderado | FN / FP con τ = 0,3 | FN / FP con τ = 0,1 |
|---|---|---|---|---|
| TF-IDF + regresión logística | 8 / 7 | — | **2** / 14 | 0 / 63 |
| BiLSTM propia (media de 3 semillas) | 11,3 / 12,7 | 11,3 / 12,7 | — | — |
| MrBERT-biomed | 5 / 1 | 5 / 2 | 4 / 2 | 4 / 8 |

En test hay 323 frases riesgosas y 183 informativas.

**La pérdida ponderada no cambió nada**, y la razón es teórica. Los pesos multiplican la pérdida de cada ejemplo, así que solo mueven el mínimo cuando el modelo no puede acertar todos los ejemplos de entrenamiento a la vez y debe elegir cuáles sacrificar. Aquí ese conflicto no existe: la BiLSTM y MrBERT llevan la pérdida de train casi a cero (0,007 y 0,0009, sección 6), es decir, separan el train por completo. Multiplicar por 2 un error que ya es casi nulo no cambia la solución. Además, las clases están balanceadas (37 % informativas), así que tampoco hay un desbalance que corregir. En la BiLSTM, la diferencia entre semillas (de 6 a 21 FN) es mucho mayor que el efecto de los pesos: es el problema de los mínimos locales de la sección 8.

**El umbral solo funciona si las probabilidades están calibradas.** La regresión logística regularizada da probabilidades moderadas: con τ = 0,3 los FN bajan de 8 a 2, a cambio de 7 FP más. MrBERT, en cambio, está **sobreconfiado**. Es la otra cara de la entropía cruzada (que premia la confianza, como se vio arriba) llevada casi a cero: sus probabilidades son casi siempre 0 o 1. Sus errores tienen P(riesgosa) cercana a 0, y ni con τ = 0,1 los recupera. Esos errores no se arreglan moviendo el umbral. Son casos sutiles que el modelo no distingue, o etiquetas dudosas: "tú no tienes riesgo" está etiquetada como informativa, y el modelo ponderado la marca, posiblemente con razón.

**Conclusión:** se mantiene MrBERT-biomed sin ponderar y con τ = 0,5. Para mejorar la sensibilidad, lo que falta son datos de los casos difíciles: los 200 desacuerdos que revisará el equipo. Calibrar el modelo, por ejemplo con *temperature scaling* sobre validación, haría útil el umbral y queda como trabajo futuro.

## 8. Descenso del gradiente y tasa de aprendizaje

El gradiente de la pérdida indica, para cada peso, en qué dirección aumenta más la pérdida. El descenso del gradiente da un paso pequeño en la dirección contraria:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla_{\mathbf{w}} L$$

donde $\eta$ es la tasa de aprendizaje. Todos los modelos del proyecto se entrenan así, por lotes (descenso del gradiente estocástico), con el optimizador AdamW, que adapta el paso de cada peso según la historia de sus gradientes. La tasa de aprendizaje es la decisión que más cambia entre modelos, y cada valor se justifica por el punto de partida:

| Modelo | Tasa de aprendizaje | Punto de partida |
|---|---|---|
| BiLSTM propia | 2 × 10⁻³ | Pesos aleatorios: hay que recorrer mucha distancia. |
| MrBERT-biomed y MrBERT-es | 3 × 10⁻⁵ | Pesos preentrenados: pasos grandes destruirían lo que ya saben (olvido catastrófico). Se usa un calentamiento del 10 % de los pasos, en que la tasa sube desde cero, para que la cabeza nueva, que parte de pesos aleatorios, no mande gradientes grandes al resto del modelo. |
| ALIA y e5-small (RAG) | 2 × 10⁻⁵ | Igual que el anterior, también con un calentamiento del 10 %. |

El curso compara el gradiente con las perillas de una máquina: indica cuánto cambia la salida al girar cada perilla. Cuando el gradiente es muy grande, un solo paso puede sacar al modelo de la zona buena. Por eso todos los entrenamientos **recortan la norma del gradiente** a 1: la red propia con `clip_grad_norm_` y los transformers con el valor por defecto de `Trainer`. El registro de MrBERT-biomed muestra cuándo actúa el recorte:

| Paso | 20 | 80 | 120 | 140 | 160 | 200 |
|---|---|---|---|---|---|---|
| Norma del gradiente | 7,9 | 8,7 | 1,4 | 0,87 | 0,03 | 0,02 |
| ¿Se recorta? | Sí | Sí | Sí | No | No | No |

El recorte solo actúa en la primera mitad del ajuste. Después, el modelo ya está cerca de un mínimo, la pendiente es casi plana y los pasos son pequeños por sí solos.

**Mínimos locales y semillas.** El descenso del gradiente garantiza bajar, pero no llegar al mínimo global: en una red con capas ocultas, la función de pérdida tiene muchos mínimos locales, y a cuál se llega depende del punto de partida. Por eso la red propia se entrenó con **3 semillas** (3 inicializaciones aleatorias distintas). En test, sus F1 son 0,895, 0,900 y 0,912, es decir, 0,902 ± 0,009. La regresión logística de la sección 3, que es convexa, da el mismo resultado siempre.

## 9. Retropropagación y regla de la cadena

Una red se puede escribir como una sola función compuesta, capa tras capa. La retropropagación calcula el gradiente de la pérdida respecto a cada peso con la regla de la cadena, empezando por la salida y **reutilizando** las derivadas ya calculadas en las capas posteriores, en lugar de recalcularlas para cada peso. En la práctica, PyTorch la ejecuta de forma automática (`perdida.backward()`).

El proyecto tuvo que usar la regla de la cadena de forma explícita en un caso: el entrenamiento de ALIA en Colab. Con lotes de 32 tripletas de 384 tokens, ModernBERT no cabía en los 15 GB de la GPU T4. La solución fue `CachedMultipleNegativesRankingLoss` (GradCache), que separa el gradiente en dos factores:

$$\frac{\partial L}{\partial \theta} = \sum_i \frac{\partial L}{\partial \mathbf{e}_i} \cdot \frac{\partial \mathbf{e}_i}{\partial \theta}$$

1. Calcula los embeddings $\mathbf{e}_i$ de todo el lote sin guardar el grafo, y con ellos la pérdida y su gradiente respecto a cada embedding, $\partial L / \partial \mathbf{e}_i$. Es barato, porque los embeddings son vectores pequeños.
2. Vuelve a pasar el lote en trozos de 8 y, para cada trozo, retropropaga ese gradiente guardado a través del modelo, $\partial \mathbf{e}_i / \partial \theta$.

El resultado es el mismo gradiente que con el lote completo de 32, pero con la memoria de un lote de 8. Así ALIA y e5-small pudieron compararse con los mismos hiperparámetros.

## 10. Fine-tuning QLoRA del generador (Fase 4)

El ajuste del generador sigue el mismo esquema de pérdida, gradiente y retropropagación, con una restricción: no se actualizan los pesos $W$ del modelo base, que quedan congelados y cuantizados a 4 bits. Cada matriz ajustada se reemplaza por $W + BA$, donde $A$ y $B$ son matrices pequeñas (de rango bajo). La retropropagación atraviesa todo el modelo, pero solo $A$ y $B$ reciben la actualización del descenso del gradiente. La pérdida será la entropía cruzada sobre el siguiente token de las explicaciones de referencia: la misma función de la sección 7, aplicada a un vocabulario de más de cien mil clases (unos 152 mil tokens si el modelo base es Qwen2.5) en lugar de 3. Se calcula solo sobre los tokens de la respuesta, no sobre los del prompt, porque el modelo debe aprender a explicar, no a repetir los datos de entrada.

Por cada matriz ajustada de tamaño $d \times k$, LoRA de rango $r$ agrega $r(d + k)$ parámetros entrenables, en lugar de los $d \cdot k$ de la matriz completa. Con $d = k = 2.048$ y $r = 16$, son 65.536 parámetros frente a 4,2 millones, un 1,6 %. Como $B$ empieza en cero, $W + BA = W$ al inicio: el modelo parte exactamente del base. Por eso la tasa de aprendizaje típica de LoRA (2 × 10⁻⁴) puede ser unas 10 veces mayor que la del ajuste fino completo de la sección 8.

### 10.1 Resultado medido

**Conteo de parámetros** (Qwen2.5-3B-Instruct, 36 capas, LoRA con r = 16 en las 7 matrices lineales de cada capa):

| Matriz | Tamaño d × k | Matrices | LoRA: r(d + k) × matrices | Matriz completa: d·k × matrices |
|---|---|---|---|---|
| q_proj, o_proj | 2.048 × 2.048 | 36 cada una | 2.359.296 cada una | 150.994.944 cada una |
| k_proj, v_proj | 256 × 2.048 | 36 cada una | 1.327.104 cada una | 18.874.368 cada una |
| gate_proj, up_proj | 11.008 × 2.048 | 36 cada una | 7.520.256 cada una | 811.597.824 cada una |
| down_proj | 2.048 × 11.008 | 36 | 7.520.256 | 811.597.824 |
| **Total** | | 252 | **29.933.568** | 2.774.532.096 |

Se entrena cerca del **1 %** de los ~3.090 millones de parámetros del modelo, y el 1,1 % de los de sus capas lineales. Las `k_proj` y `v_proj` son más chicas (256 filas) porque Qwen2.5 comparte las claves y valores entre grupos de cabezas de atención. Un detalle de medición: el total que imprime PyTorch (1.729 millones) está subestimado, porque cada par de pesos de 4 bits se guarda empaquetado en un solo byte.

**Curva de aprendizaje** (555 ejemplos, 2 épocas, lote efectivo de 16, 70 pasos):

| Paso | Época | Pérdida en train | Pérdida en validación | Exactitud por token (val) |
|---|---|---|---|---|
| 5 | 0,14 | 0,793 | — | — |
| 25 | 0,72 | 0,561 | 0,607 | 82,4 % |
| 50 | 1,43 | 0,439 | 0,547 | 83,8 % |
| 70 | 2,00 | 0,423 | 0,538 | 84,0 % |

Hay tres contrastes con los modelos anteriores:
- **No hay sobreajuste visible.** La pérdida de validación baja en cada evaluación, aunque cada vez menos (−0,060 y luego −0,009). La red propia de la sección 6, en cambio, llevaba la pérdida de train casi a cero mientras la validación oscilaba. Aquí el modelo solo puede mover el 1 % de sus parámetros, y eso actúa como una regularización por sí misma.
- **El recorte del gradiente nunca actuó:** la norma se mantuvo entre 0,24 y 0,39. En MrBERT llegaba a 8,7 al inicio (sección 8), porque la cabeza nueva empezaba aleatoria. Con LoRA, $B = 0$ al inicio, así que el primer paso parte de la solución del modelo base y los gradientes son pequeños.
- **La pérdida de un modelo de lenguaje no llega a cero:** 0,42 en train equivale a una probabilidad media de ~0,66 para el token correcto ($e^{-0{,}42}$). Hay muchas formas válidas de escribir la misma explicación, y la entropía cruzada mide el acierto palabra por palabra contra una sola de ellas. Por eso el modelo se evalúa con métricas de la tarea (cobertura, fidelidad, legibilidad), igual que los clasificadores se entrenan con entropía cruzada y se evalúan con F1 (sección 7).

**Resultado en la tarea** (150 casos de test): la cobertura de exámenes pasa de 68,7 % a 99,3 %, el estado consistente de 69,7 % a 99,4 % y mencionar el valor del paciente de 10,3 % a 70,7 %. El detalle está en `PLAN.md`. Este último número muestra algo del aprendizaje supervisado: **el estudiante aprende la distribución de sus datos, defectos incluidos.** El profesor que aportó la mitad del train (Qwen2.5-14B) menciona el valor solo en el 41,5 % de sus respuestas, y el promedio ponderado de los tres profesores (~66 %) predice casi exactamente lo que logra el estudiante.

## Resumen

| Principio del curso | Dónde aparece en LetraDeMédico | Evidencia |
|---|---|---|
| Asociacionismo (4 leyes de Aristóteles) | Ajuste contrastivo con negativos difíciles | hit@1 en no vistos: 84,4 % → 91,2 % (ALIA) |
| Hebb y su inestabilidad | Softmax (competencia) y coseno acotado (límite) en el RAG; weight decay (reducción) solo en guardrails | 2 de los 3 problemas resueltos en el RAG; el tercero queda pendiente |
| Neurona de McCulloch y Pitts (pesos fijos, sin aprendizaje, sin sinapsis inhibidoras) | Filtro de palabras original | F1 binario 0,219; 30 de 60 falsos positivos son el propio aviso de seguridad |
| Perceptrón, sesgo y separabilidad lineal | TF-IDF + regresión logística (8.172 parámetros) | F1 binario 0,977; falla en los casos no lineales |
| Funciones de activación | Sigmoide y tanh en la LSTM, ReLU en su cabeza, GeGLU y SiLU en MrBERT, softmax de salida | Elección justificada por capa |
| Red retroalimentada y conteo de parámetros | Cabeza de la red propia (131 neuronas) y red completa | 456.707, desglosados por capa |
| Poder de representación y sobreajuste | Red propia frente a transformers | Pérdida en train 0,007 con F1 en validación ≈ 0,93 |
| Función de pérdida (MSE, cero-uno, entropía cruzada, ponderada) | Entropía cruzada para entrenar, F1 para evaluar; experimento de pérdida ponderada frente a umbral | Ponderar no cambia nada con train separable; el umbral solo funciona en el modelo calibrado (TF-IDF: FN 8 → 2) |
| Descenso del gradiente, tasa de aprendizaje y mínimos locales | AdamW con 2 × 10⁻³ desde cero y 2 a 3 × 10⁻⁵ en ajuste fino; 3 semillas | Recorte activo en los primeros ~130 de 228 pasos; F1 de 0,902 ± 0,009 entre semillas |
| Ajuste fino eficiente: LoRA + cuantización (QLoRA) | Generador Qwen2.5-3B, 1 % de parámetros entrenables | Cobertura 68,7 % → 99,3 %; estado consistente 69,7 % → 99,4 %; validación siempre a la baja |
| Retropropagación y regla de la cadena | GradCache para entrenar ALIA en la T4; retropropagación a través del tiempo en la BiLSTM | Lote efectivo de 32 con memoria de 8 |

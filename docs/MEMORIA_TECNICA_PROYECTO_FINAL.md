# Memoria técnica — Proyecto Final IA Generativa (Evolve)

**Título:** Asistente experto documental de contratos laborales ficticios con Gemini, RAG y agentes LangGraph  
**Fecha de corte:** 27/10/2026  
**Estado:** prototipo local con evaluación técnica e incidencias documentadas. Revisión factual y jurídica exhaustiva pendiente.

## Resumen ejecutivo

Se implementó un prototipo de pregunta-respuesta especializado en dos contratos laborales ficticios y dos documentos normativos españoles aportados. Mediante recuperación vectorial con Gemini Embeddings y ChromaDB, generación con Gemini y grafos LangGraph, el sistema responde a preguntas directas, compara documentos, registra referencias a PDF y resuelve referencias conversacionales dentro de una sesión en RAM. Se creó un corpus trazable con 4 PDF (110 páginas) y 437 fragmentos.

## Objetivos y cumplimiento

| Requisito oficial del proyecto | Implementación / evidencia |
|---|---|
| Dominio experto definido | Contratos ficticios y normas laborales españolas aportadas |
| ≥3 documentos o ~20 páginas | 4 documentos, 110 páginas |
| Gemini Embeddings + ChromaDB | 437 vectores, 768 dimensiones, `gemini-embedding-2` |
| RAG Gemini | `scripts/etapa5_rag.py`, modelo `gemini-3.5-flash-lite` |
| System prompt documentado | `README.md`, sección «System prompt»; versión operativa en script |
| LangGraph condicional | Subagente en `scripts/etapa6_langgraph.py` |
| Memoria conversacional | Envolvente `scripts/etapa7_memoria.py` y `InMemorySaver` por `thread_id` |
| Chat en notebook | Sección 8.3 del `.ipynb`; activación explícita para evitar API automática |
| Cinco preguntas documentadas | Sección 6 del notebook, con fuentes y respuestas principales observadas |
| Presentación oral 5–10 minutos | Completada. Presentación y vídeo entregados por separado, fuera de GitHub. |

## Datos, procedencia y calidad

Las fuentes documentales son DOC-001 (3 páginas), DOC-002 (3), NOR-001 (97) y NOR-002 (7). Las versiones normativas son las suministradas en el corpus, y los datos originales se preservaron. Se extraen, limpian y segmentan en fragmentos con metadatos (`documento_id`, `seccion`, `pagina_inicio`, `pagina_fin`) y controles SHA256. El corpus aprobado para embeddings tiene 437 fragmentos: 14 DOC-001, 13 DOC-002, 387 NOR-001 y 23 NOR-002. Esta distribución no representa un muestreo estadístico del derecho laboral.

La vectorización utiliza `gemini-embedding-2` (768 dimensiones), con una cadena por solicitud; los embeddings se guardan en `evolve_contratos_completa_v1` de ChromaDB. Las consultas generan embeddings de preguntas con el mismo modelo. El filtrado por documento permite recuperar evidencia de cada fuente cuando la pregunta solicita una comparación. La colección no se regenera en la Etapa 9.

## Prompts y seguridad de respuestas

El prompt de sistema exige responder solo con las evidencias proporcionadas, referenciar fragmentos `[F#]`, tratar el texto recuperado como datos no confiables, distinguir documento ficticio de norma y abstenerse ante evidencia insuficiente. Se intenta restringir contenido accesorio no pedido. El sistema valida presencia y estructura de citas, pero la coincidencia formal no acredita que cada afirmación esté respaldada por ellas.

Las claves están fuera del código y no se deben mostrar ni subir al repositorio. La anonimización de textos antes de consultas es parcial; no introducir contratos reales en esta demostración. El corpus se conserva localmente y las llamadas a Gemini envían texto a un proveedor externo, sujeto a sus condiciones.

## Modelo de agente y memoria

El grafo interno de LangGraph decide si responde directamente a saludos/ayuda o si recupera documentos. En la ruta RAG, busca, valida contexto y genera con Gemini o se abstiene. El grafo exterior resuelve seguimientos como «el otro contrato» y registra historial breve, aislado por `thread_id`. `InMemorySaver` almacena estados durante el proceso actual; no hay persistencia de conversaciones al reiniciar el kernel. Un ejemplo de tres turnos consecutivos validó el cambio de DOC-001 a DOC-002 y la comparación posterior.

## Evaluación cuantitativa, hallazgos y mejoras

**Recuperación (Etapa 4):** sobre 10 preguntas simples: Hit@1 = 0,80; Hit@5 = 1,00; MRR@5 = 0,8833. No son medidas de exactitud factual. Para dos preguntas de fuentes múltiples, el sistema de búsquedas dirigidas por documento recuperó ambas evidencias en 2/2 casos con hasta 5 resultados por fuente; esa cifra no equivale al Hit@5 global.

**Pruebas de integración (hasta Etapa 8.1):** 141/141 pruebas en Windows del alumno. Validan casos cubiertos por tests, no todas las respuestas posibles.

**Validación final de entrega (Etapa 9):** 153/153 pruebas automatizadas superadas. Notebook, corpus, ChromaDB y evidencias validados mediante el preflight offline.

**Evaluación integral histórica (Etapa 8):** 11 casos; el evaluador inicial no detectó errores estructurales. La auditoría 8.1 posterior localizó 2 incidencias de trazabilidad de citas (D05, D09), y 9 casos sin incidencias estructurales adicionales detectadas. El control mejorado pasa tests; sin una nueva evaluación del LLM no debe afirmarse que los registros antiguos se corrigieron retroactivamente.

**Revisión factual preliminar:** D03 no explicó la excepción por convenio sectorial que puede ampliar el límite temporal para contratos por circunstancias de la producción; D08 incluyó fechas de firma no solicitadas y no recuperó la firma existente de DOC-001; se documentó una inconsistencia de identidad en la cláusula QUINTA de DOC-001. D05 y D09 contienen incidencias históricas de correspondencia entre texto y listado de citas. No afirmar exactitud ni legalidad del 100 %.

## Reproducibilidad del entregable

Se abre el notebook en VS Code con el `.venv` Python 3.12 existente, y se ejecuta en orden. Los primeros bloques comprueban el SHA256 y que Chroma contiene 437 registros y muestran configuración, prompt, rutas y evaluación histórica **sin llamadas Gemini**. El notebook ofrece una simulación de memoria y tres activadores para llamadas reales (cinco ejemplos, memoria de tres turnos y chat). Las consultas reales requieren clave en entorno o una solicitud privada de `getpass`. No es un notebook autónomo separado del proyecto, ya que importa los módulos y los vectores existentes.

## Validez, limitaciones y trabajo futuro

El sistema está limitado a versiones aportadas del BOE, no valida vigencia jurídica externa, no admite datos personales reales, no garantiza ausencia de alucinaciones, ni persiste historial después de reiniciar Python. Los casos de evaluación son pocos, por lo que no permiten generalización. Se recomienda revisión humana de cada afirmación, incremento del banco de pruebas, validación semántica de citas, un plan de persistencia segura y revisión de condiciones de uso de APIs si se escala.

## Conclusión

La arquitectura ofrece una demostración técnica reproducible de Gemini, RAG, ChromaDB, LangGraph y memoria temporal. La presentación final debe enfatizar la trazabilidad, la evaluación iterativa y los límites observados, no equiparar cobertura de pruebas o citas formales con certeza jurídica.

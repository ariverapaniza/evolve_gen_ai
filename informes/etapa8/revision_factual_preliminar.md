# Proyecto Final de IA Generativa — Evolve
## Etapa 8.2 — Revisión documental preliminar de respuestas y citas

**Fecha de revisión:** 27 de octubre de 2026  
**Naturaleza:** revisión técnica y documental asistida; **no constituye revisión jurídica profesional ni firma de revisión humana definitiva**.

### 1. Alcance y evidencia consultada

- Salida de la evaluación integral enviada por el estudiante: 11 casos D01–D11; 132/132 pruebas en la ejecución original.
- Salida de la auditoría 8.1 enviada por el estudiante: 141/141 pruebas, **9 controles estructurales sin incidencias detectadas** y **2 incidencias D05/D09**.
- PDF originales de los **dos contratos ficticios** (DOC-001 y DOC-002), con revisión visual de las páginas 2 y 3 de ambos.
- Corpus depurado `chunks_para_embeddings.jsonl` con 437 fragmentos de las cuatro fuentes. Las comprobaciones de NOR-001/NOR-002 aquí se realizan **contra el corpus indexado**, no implican certificación de vigencia normativa actual.
- Los identificadores `[F#]` son relativos **a cada consulta**, no son identificadores globales ni persistentes de los fragmentos.

### 2. Resumen de hallazgos

| Caso | Comprobación documental preliminar | Estado / acción |
|---|---|---|
| D01 | Saludo y descripción académica del servicio. No requiere evidencias de los PDF. | **Sin incidencia detectada** |
| D02 | `DOC-001`, p. 2, cláusula PRIMERA: cargo CDAIO y cuatro funciones indicadas. | **Coincide en lo principal**; verificar cada afirmación secundaria si se reproduce en la memoria final |
| D03 | `NOR-001`, artículo 15, fragmentos `NOR-001-00055` y `00056`: regla general de contratación indefinida y supuestos temporales. | **Matiz sustantivo omitido**: el plazo general de 6 meses por circunstancias de la producción puede ampliarse **hasta un año por convenio colectivo sectorial**, según el propio corpus. La respuesta de D03 no lo aclaró. No presentar el límite de seis meses como absoluto. |
| D04 | `NOR-002`, artículo 1, fragmento `NOR-002-00003`: concepto de alta dirección, exclusión y referencia al sector público. | **Coincide con el fragmento indexado**; revisar formalmente la fuente normativa antes de conclusiones jurídicas |
| D05 | `DOC-002`, p. 2, cláusula SEXTA: **22 días laborables** por año natural. | **Dato principal verificado**; **incidencia de citas histórica**: la respuesta usa `[F1, F4]` pero solo apareció `[F2]` en el listado del informe original. El auditor 8.1 la detectó. |
| D06 | `DOC-001`, p. 2, cláusula TERCERA: vigencia desde **1 de abril de 2026**. | **Dato principal verificado** |
| D07 | `DOC-002`, p. 2, cláusula TERCERA: vigencia desde **15 de abril de 2026**. Resuelto usando contexto de D06. | **Dato y seguimiento conversacional verificados** |
| D08 | Cláusulas TERCERA de ambos PDF, p. 2: las dos fechas son correctas. | **Comparación principal correcta**. Observación: `DOC-001`, p. 3, **sí contiene fecha de firma: 20 de marzo de 2026**. La frase «no aparece en los fragmentos recuperados» no equivale a «no aparece en el documento». El detalle de las firmas no era solicitado. |
| D09 | `DOC-002`, p. 2: **22 días laborables**. `NOR-001`, art. 38, fragmento `NOR-001-00136`: **mínimo 30 días naturales**, vacaciones retribuidas. | **Comparación principal consistente con las fuentes**. No afirmar equivalencia automática entre días laborables y naturales. **Incidencia de cita histórica**: `[F7]` está en la respuesta, pero no en las referencias impresas. |
| D10 | El contrato ficticio de alta dirección no contiene un número de satélites lanzados. | **Abstención apropiada para el corpus examinado**; evitar inferir que la empresa ficticia nunca lanzó satélites en el mundo real. |
| D11 | «¿Y qué indica el otro contrato?» en sesión independiente sin antecedente. | **Aclaración apropiada; no se mezclaron sesiones** |

### 3. Una inconsistencia interna de los datos de origen

En `DOC-001`, p. 2, cláusula QUINTA (Retribución), se identifica como receptor de un pago a una persona **distinta del directivo nombrado en el encabezamiento y en la firma del contrato**. Es una inconsistencia del **PDF ficticio original** y no debe corregirse silenciosamente en el corpus. Si se pregunta sobre esa cláusula, el asistente debería señalar la discrepancia de identidad y abstenerse de atribuir inequívocamente esa información al directivo sin aclaración. Por privacidad y relevancia académica, este informe omite los datos bancarios e identificativos completos.

### 4. Interpretación correcta de las métricas

1. **141/141 pruebas `unittest`:** confirma los comportamientos técnicos cubiertos por esos tests; no mide precisión jurídica.
2. **11/11 ejecuciones originales:** las rutas y controles de entonces finalizaron sin fallos registrados, **antes** de mejorar el analizador de citas agrupadas.
3. **Auditoría 8.1 aplicada a respuestas históricas:** **9/11** casos sin errores estructurales adicionales detectados y **2/11** con incidencias de trazabilidad (D05, D09). **No reclasificar retroactivamente como 11/11** sin una nueva ejecución de esas respuestas y su revisión.
4. El analizador de citas y el *system prompt* actualizados tienen pruebas de regresión. Esto indica que el software está mejor preparado **para futuras generaciones**, pero **no corrige retroactivamente** el registro que ya se guardó.
5. **Los 437 embeddings y ChromaDB no necesitan regenerarse.**

### 5. Decisiones recomendadas antes de redactar el notebook y la memoria

- **Conservar** `informes/etapa8/evaluacion_integral.json` y `informes/etapa8/auditoria_citas_v81.json` como evidencia de antes/después del auditor.
- **Documentar como hallazgos** D03 (matiz del límite temporal), D05 y D09 (trazabilidad) y D08 (respuesta excesivamente amplia / falta de recuperación de firma).
- **No afirmar** precisión factual ni jurídica del 100 %. El banco de 11 casos es pequeño y exploratorio.
- En la documentación académica, especificar que las normas se analizan **según las versiones aportadas**, sin revisión externa automática de vigencia.
- Si se desea medir una mejora cuantitativa posterior, ejecutar una **nueva evaluación completa en una copia separada** (nueva fecha y modelo registrado), que sí utilizará llamadas a Gemini. No sobrescribir la línea base.

### 6. Revisión definitiva pendiente

Esta revisión permite preparar la **Etapa 9** con resultados y limitaciones veraces, pero no sustituye la contrastación exhaustiva de **cada frase** de las nueve respuestas documentales con sus respectivos documentos y disposiciones. Antes de entregar un trabajo que haga afirmaciones jurídicas, completar esa verificación humana y citar con precisión la versión normativa utilizada.

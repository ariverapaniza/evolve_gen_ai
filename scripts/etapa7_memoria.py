"""Etapa 7 Evolve | Memoria conversacional por sesión mediante LangGraph InMemorySaver.

Reutiliza SIN modificar el grafo real de la Etapa 6 (que a su vez integra RAG v5.1).
CLI: check | simular | chat --sesion sesion-1
No escribe claves ni historial en disco; NO modifica ChromaDB ni los PDF.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import unicodedata
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import etapa6_langgraph as e6

MAX_HISTORIAL = 8
MAX_PREGUNTA = 800
VALIDAS = ("DOC-001", "DOC-002", "NOR-001", "NOR-002")


def normalizar(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.casefold())
                   if unicodedata.category(c) != "Mn")


def combinar_historial(anterior: list[dict], nuevos: list[dict]) -> list[dict]:
    """Mantiene como máximo ocho turnos sin perder la semántica de agregar turnos."""
    return (list(anterior or []) + list(nuevos or []))[-MAX_HISTORIAL:]


def tema_de(texto: str) -> str | None:
    q = normalizar(texto)
    if ("fecha" in q and any(t in q for t in ("inicio", "vigencia", "comienzo"))) or "cuando empieza" in q or "cuando comienza" in q:
        return "fecha_inicio"
    for fragmentos, nombre in (
        (("vacacion",), "vacaciones"),
        (("periodo de prueba",), "periodo_prueba"),
        (("no competencia", "no concurrencia"), "no_competencia"),
        (("retribucion", "salario", "sueldo", "remuneracion"), "retribucion"),
        (("cargo", "funciones", "puesto de trabajo"), "cargo_funciones"),
        (("jornada", "horas de trabajo"), "jornada"),
        (("duracion", "vigencia"), "duracion"),
    ):
        if any(s in q for s in fragmentos):
            return nombre
    return None


def documento_mencionado(texto: str) -> str | None:
    q = normalizar(texto)
    if "doc-001" in q or "contrato de alta direccion" in q or "contrato del directivo" in q:
        return "DOC-001"
    if "doc-002" in q or "contrato indefinido" in q or "contrato indeterminado" in q:
        return "DOC-002"
    if "nor-001" in q or "estatuto de los trabajadores" in q:
        return "NOR-001"
    if "nor-002" in q or "real decreto 1382/1985" in q:
        return "NOR-002"
    return None


def nombre_documento(doc: str) -> str:
    return {
        "DOC-001": "contrato de alta dirección",
        "DOC-002": "contrato indefinido",
        "NOR-001": "Estatuto de los Trabajadores",
        "NOR-002": "Real Decreto 1382/1985",
    }[doc]


def pregunta_del_tema(tema: str, doc: str) -> str:
    nombre = nombre_documento(doc)
    if tema == "fecha_inicio":
        return f"¿En qué fecha se inicia la vigencia del {nombre}?"
    if tema == "vacaciones":
        return f"¿Qué establece el {nombre} sobre las vacaciones anuales?"
    if tema == "cargo_funciones":
        return f"¿Qué cargo, puesto o funciones establece el {nombre}?"
    if tema == "duracion":
        return f"¿Cuál es la duración y vigencia del {nombre}?"
    if tema == "periodo_prueba":
        return f"¿Qué indica el {nombre} sobre el período de prueba?"
    if tema == "retribucion":
        return f"¿Qué indica el {nombre} sobre la retribución?"
    if tema == "jornada":
        return f"¿Qué indica el {nombre} sobre la jornada laboral?"
    if tema == "no_competencia":
        return f"¿Qué indica el {nombre} sobre el pacto de no competencia?"
    raise ValueError("Tema de conversación no compatible con reformulación segura")


def documento_principal(turno: dict) -> str | None:
    foco = turno.get("fuente_foco")
    if foco in ("DOC-001", "DOC-002", "NOR-001", "NOR-002"):
        return foco
    fuentes = turno.get("fuentes", [])
    return fuentes[0] if len(fuentes) == 1 else None


def resolver_seguimiento(pregunta: str, historial: list[dict], explicitas: list[str] | None = None) -> dict:
    """Resolver referencias explícitas, sin adivinar hechos ni emplear el LLM."""
    if not isinstance(pregunta, str) or not pregunta.strip() or len(pregunta) > MAX_PREGUNTA:
        raise ValueError("La pregunta debe contener entre 1 y 800 caracteres")
    fuentes = list(explicitas or [])
    if len(set(fuentes)) != len(fuentes) or any(x not in VALIDAS for x in fuentes):
        raise ValueError("Fuentes explícitas inválidas o duplicadas")
    q = normalizar(pregunta).strip()
    q = q.strip('¿?¡!., ')
    turnos = [x for x in historial if x.get("ruta") not in ("directa", "aclaracion")]
    anterior = turnos[-1] if turnos else None
    tema = tema_de(pregunta)
    fuente_mencionada = documento_mencionado(pregunta)

    # Si el usuario precisa fuentes, la selección explícita prima sobre la memoria.
    if fuentes:
        return {"pregunta_resuelta": pregunta, "fuentes_resueltas": fuentes,
                "usa_memoria": False, "motivo": "", "tema": tema or (anterior or {}).get("tema")}

    # Comparación de ambos contratos después de hablar de alguno.
    solicita_comparar = any(x in q for x in ("compara", "comparar", "comparalas", "comparalos", "diferencia entre"))
    referido_ambos = any(x in q for x in ("ambos", "ambas", "dos contratos", "entre los dos"))
    if solicita_comparar and referido_ambos and anterior and not fuente_mencionada:
        tema_final = tema or anterior.get("tema")
        if tema_final == "fecha_inicio":
            texto = "Compara las fechas de inicio de los dos contratos"
        elif tema_final == "vacaciones":
            texto = "Compara las vacaciones de los dos contratos"
        elif tema_final in ("periodo_prueba", "retribucion", "jornada", "duracion", "no_competencia", "cargo_funciones"):
            texto = "Compara la información sobre " + {
                "periodo_prueba": "el período de prueba", "retribucion": "la retribución",
                "jornada": "la jornada laboral", "duracion": "la duración",
                "no_competencia": "el pacto de no competencia", "cargo_funciones": "el puesto y las funciones",
            }[tema_final] + " de los dos contratos"
        else:
            return {"pregunta_resuelta": pregunta, "fuentes_resueltas": [], "usa_memoria": False,
                    "motivo": "Indica qué aspecto de los dos contratos deseas comparar.", "tema": None}
        return {"pregunta_resuelta": texto, "fuentes_resueltas": ["DOC-001", "DOC-002"],
                "usa_memoria": True, "motivo": "", "tema": tema_final}

    # Un seguimiento comparativo sin tema conocido no debe deducirse sin base.
    otro = "otro contrato" in q or q in ("y el otro", "el otro", "y que dice el otro")
    if otro:
        if not anterior:
            mensaje = "¿A qué contrato y sobre qué asunto te refieres?"
        else:
            fuente = documento_principal(anterior)
            tema_final = tema or anterior.get("tema")
            if fuente not in ("DOC-001", "DOC-002") or not tema_final:
                mensaje = "Indica el contrato de referencia y el asunto que deseas comparar."
            else:
                destino = "DOC-002" if fuente == "DOC-001" else "DOC-001"
                return {"pregunta_resuelta": pregunta_del_tema(tema_final, destino),
                        "fuentes_resueltas": [destino], "usa_memoria": True,
                        "motivo": "", "tema": tema_final}
        return {"pregunta_resuelta": pregunta, "fuentes_resueltas": [], "usa_memoria": False,
                "motivo": mensaje, "tema": tema}

    # Referencia al mismo documento: se toma el tema nuevo explícito, si existe.
    seguimiento = bool(re.match(r"^(?:y |y\b|¿?y\b|¿?y en |¿?y cual |¿?y que |¿?y respecto)", q))
    if seguimiento and not fuente_mencionada:
        if anterior and documento_principal(anterior):
            origen = documento_principal(anterior)
            tema_final = tema or anterior.get("tema")
            if tema_final:
                return {"pregunta_resuelta": pregunta_del_tema(tema_final, origen),
                        "fuentes_resueltas": [origen], "usa_memoria": True,
                        "motivo": "", "tema": tema_final}
        return {"pregunta_resuelta": pregunta, "fuentes_resueltas": [], "usa_memoria": False,
                "motivo": "Indica a qué documento y asunto te refieres.", "tema": tema}

    # Pregunta autónoma: nunca acarrear filtros de una pregunta anterior.
    if fuente_mencionada and not any(x in q for x in ("compara", "comparar", "ambos", "ambas")):
        fuentes = [fuente_mencionada]
    return {"pregunta_resuelta": pregunta, "fuentes_resueltas": fuentes,
            "usa_memoria": False, "motivo": "", "tema": tema}


class EstadoMemoria(TypedDict, total=False):
    pregunta: str
    fuentes_explicitas: list[str]
    top_k: int
    modelo: str
    pregunta_resuelta: str
    fuentes_resueltas: list[str]
    tema: str | None
    usa_memoria: bool
    motivo: str
    respuesta: str
    ruta: str
    recorrido: list[str]
    referencias: list[dict]
    historial: Annotated[list[dict], combinar_historial]


def crear_grafo(*, ejecutor=None, checkpointer=None):
    """Grafo externo con estado persistido por thread_id; etapa6 es el subagente."""
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    saver = checkpointer if checkpointer is not None else InMemorySaver()
    base = {"agente": None}

    def preparar(estado: EstadoMemoria) -> dict:
        nuevo = resolver_seguimiento(
            estado["pregunta"], estado.get("historial", []), estado.get("fuentes_explicitas", []))
        return {**nuevo, "recorrido": ["preparar_contexto"], "respuesta": "", "referencias": [], "ruta": ""}

    def decidir(estado: EstadoMemoria) -> str:
        return "aclarar" if estado.get("motivo") else "ejecutar"

    def aclarar(estado: EstadoMemoria) -> dict:
        return {"respuesta": estado["motivo"], "ruta": "aclaracion",
                "recorrido": estado["recorrido"] + ["pedir_aclaracion"]}

    def ejecutar(estado: EstadoMemoria) -> dict:
        if ejecutor is not None:
            salida = ejecutor(estado["pregunta_resuelta"], estado["fuentes_resueltas"],
                              estado.get("top_k", 5), estado.get("modelo", e6.rag.DEFAULT_MODEL))
        else:
            if base["agente"] is None:
                base["agente"] = e6.crear_grafo()
            salida = base["agente"].invoke({
                "pregunta": estado["pregunta_resuelta"],
                "fuentes_explicitas": estado["fuentes_resueltas"],
                "top_k": estado.get("top_k", 5),
                "modelo": estado.get("modelo", e6.rag.DEFAULT_MODEL),
            })
        evidencias = salida.get("evidencias", [])
        citas = salida.get("citas", [])
        referencias = []
        for n in sorted(set(citas)):
            if isinstance(n, int) and 1 <= n <= len(evidencias):
                m = evidencias[n-1]["metadata"]
                referencias.append({"ref": f"F{n}", "id_chunk": evidencias[n-1].get("id", ""),
                                    "documento_id": m["documento_id"], "seccion": m["seccion"],
                                    "pagina_inicio": m["pagina_inicio"], "pagina_fin": m["pagina_fin"]})
        return {"respuesta": salida.get("respuesta", ""), "ruta": salida.get("ruta", ""),
                "recorrido": estado["recorrido"] + salida.get("recorrido", []) + ["registrar_turno"],
                "referencias": referencias}

    def registrar(estado: EstadoMemoria) -> dict:
        focus = documento_mencionado(estado["pregunta_resuelta"])
        if len(estado.get("fuentes_resueltas", [])) == 1:
            focus = estado["fuentes_resueltas"][0]
        if not focus:
            docs = {m["documento_id"] for m in estado.get("referencias", [])
                    if m["documento_id"] in ("DOC-001", "DOC-002")}
            if len(docs) == 1:
                focus = next(iter(docs))
        turno = {
            "pregunta": estado["pregunta"],
            "pregunta_resuelta": estado.get("pregunta_resuelta", estado["pregunta"]),
            "respuesta": estado.get("respuesta", ""),
            "ruta": estado.get("ruta", ""),
            "tema": estado.get("tema"),
            "fuente_foco": focus,
            "fuentes": estado.get("fuentes_resueltas", []),
            "usa_memoria": bool(estado.get("usa_memoria")),
            "referencias": estado.get("referencias", []),
        }
        return {"historial": [turno]}

    grafo = StateGraph(EstadoMemoria)
    grafo.add_node("preparar_contexto", preparar)
    grafo.add_node("pedir_aclaracion", aclarar)
    grafo.add_node("ejecutar_agente", ejecutar)
    grafo.add_node("registrar_turno", registrar)
    grafo.add_edge(START, "preparar_contexto")
    grafo.add_conditional_edges("preparar_contexto", decidir, {
        "aclarar": "pedir_aclaracion", "ejecutar": "ejecutar_agente",
    })
    grafo.add_edge("pedir_aclaracion", "registrar_turno")
    grafo.add_edge("ejecutar_agente", "registrar_turno")
    grafo.add_edge("registrar_turno", END)
    return grafo.compile(checkpointer=saver)


def preguntar(grafo, *, sesion: str, texto: str, fuentes: list[str] | None = None,
             top_k: int = 5, modelo: str | None = None) -> dict:
    if not isinstance(sesion, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", sesion):
        raise ValueError("--sesion debe contener 1-64 caracteres alfanuméricos, _ o -")
    if not 1 <= top_k <= 5:
        raise ValueError("--top-k debe estar entre 1 y 5")
    return grafo.invoke({
        "pregunta": texto, "fuentes_explicitas": list(fuentes or []),
        "top_k": top_k, "modelo": modelo or e6.rag.DEFAULT_MODEL,
    }, config={"configurable": {"thread_id": sesion}})


def imprimir(res: dict) -> None:
    print("RUTA:", res.get("ruta", "?"))
    print("PREGUNTA INTERPRETADA:", res.get("pregunta_resuelta", ""))
    print("DOCUMENTOS:", res.get("fuentes_resueltas", []))
    print("SE UTILIZÓ MEMORIA:", "sí" if res.get("usa_memoria") else "no")
    print("NODOS:", " -> ".join(res.get("recorrido", [])))
    for m in res.get("referencias", []):
        ref = m.get("ref", "F?")
        print(f"  [{ref}] {m['documento_id']} | {m['seccion']} | PDF pág. {m['pagina_inicio']}-{m['pagina_fin']}")
    print("RESPUESTA:\n" + res.get("respuesta", ""))
    print(f"TURNOS GUARDADOS EN RAM: {len(res.get('historial', []))}")


def ejemplo_ficticio(pregunta, fuentes, top_k, modelo):
    """Doble de prueba, NO es Gemini ni consulta ChromaDB."""
    ruta, _ = e6.ruta_inicial(pregunta, fuentes)
    return {"ruta": ruta, "recorrido": ["clasificar", "simulacion_sin_api"],
            "respuesta": "SIMULACIÓN LOCAL — consulta: " + pregunta,
            "citas": [], "evidencias": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="accion", required=True)
    sub.add_parser("check", help="Comprueba las rutas y la base Chroma sin usar API")
    sub.add_parser("simular", help="Memoria de tres turnos y aislamiento de sesiones, SIN Gemini")
    chat = sub.add_parser("chat", help="Conversación interactiva. La memoria dura hasta cerrar el proceso")
    chat.add_argument("--sesion", default="evolve_demo")
    chat.add_argument("--top-k", type=int, default=5)
    chat.add_argument("--modelo", default=e6.rag.DEFAULT_MODEL)
    args = ap.parse_args()
    try:
        if args.accion == "check":
            from langgraph.checkpoint.memory import InMemorySaver
            g = crear_grafo(checkpointer=InMemorySaver(), ejecutor=ejemplo_ficticio)
            result = preguntar(g, sesion="comprobacion", texto="Hola")
            assert result["ruta"] == "directa" and len(result["historial"]) == 1
            col = e6.rag.preflight()
            print("LANGGRAPH: OK | InMemorySaver y thread_id")
            print(f"CHROMA: OK | {col.count()} embeddings | SIN LLAMADAS API")
            print("GEMINI_API_KEY:", "configurada" if e6.rag.e3.clave_configurada() else "pendiente")
            print("RESULTADO: PREFLIGHT MEMORIA CORRECTO")
            return
        if args.accion == "simular":
            g = crear_grafo(ejecutor=ejemplo_ficticio)
            turnos = [
                "¿Cuál es la fecha de inicio del contrato de alta dirección?",
                "¿Y qué indica el otro contrato?",
                "Compara ambas fechas de inicio",
            ]
            for i, pregunta in enumerate(turnos, 1):
                result = preguntar(g, sesion="alumno", texto=pregunta)
                print(f"\nTURNO {i} — SIMULACIÓN SIN GEMINI")
                imprimir(result)
            aislado = preguntar(g, sesion="otra_sesion", texto="¿Y qué indica el otro contrato?")
            print("\nSESIÓN INDEPENDIENTE:", aislado["ruta"], "|", aislado["respuesta"])
            assert aislado["ruta"] == "aclaracion"
            print("RESULTADO: MEMORIA Y AISLAMIENTO COMPROBADOS SIN API")
            return
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", args.sesion):
            raise ValueError("Identificador de sesión inválido")
        g = crear_grafo()
        print("CHAT Evolve — Memoria en RAM por thread_id:", args.sesion)
        print("Introduce una pregunta. Escribe 'salir' para cerrar y descartar el historial.")
        print("NOTA: las preguntas documentales usan Gemini y pueden consumir cuota de API.")
        while True:
            try:
                texto = input("\nTÚ> ").strip()
            except EOFError:
                break
            if texto.casefold() in {"salir", "exit", "quit"}:
                break
            if not texto:
                continue
            try:
                salida = preguntar(g, sesion=args.sesion, texto=texto, top_k=args.top_k, modelo=args.modelo)
                imprimir(salida)
                print("NOTA: contenido jurídico académico; revisar afirmaciones contra los PDF.")
            except Exception as exc:
                print(f"ERROR DEL TURNO: {type(exc).__name__}: {exc}", file=sys.stderr)
                print("La sesión sigue abierta; puedes formular otra pregunta.", file=sys.stderr)
        print("CHAT CERRADO: historial conversacional en RAM descartado.")
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

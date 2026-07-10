"""
OpenWikiTool - Integracion de langchain-ai/openwiki en LegoGPT.

Referencia: https://github.com/langchain-ai/openwiki

Instalacion:
    pip install langchain-community wikipedia langchain-core

Uso:
    from src.tools.openwiki_tool import OpenWikiTool, get_lego_wiki_context

    tool = OpenWikiTool(lang="es", top_k_results=2)
    resultado = tool.search("LEGO Technic")
    contexto = get_lego_wiki_context("Mindstorms")
    lc_tool = tool.as_langchain_tool()
"""
import logging
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Importacion de WikipediaAPIWrapper.
# langchain-community esta siendo deprecado. Cuando langchain-wikipedia
# (paquete standalone) este disponible se usara preferentemente.
# Ref: https://github.com/langchain-ai/openwiki
# ---------------------------------------------------------------------------
_LANGCHAIN_AVAILABLE = False
WikipediaAPIWrapper = None

# Intentar primero el paquete standalone (futuro)
try:
    from langchain_wikipedia import WikipediaAPIWrapper  # type: ignore[no-redef]
    _LANGCHAIN_AVAILABLE = True
    logger.debug("Usando langchain-wikipedia (paquete standalone)")
except ImportError:
    pass

# Fallback a langchain-community (actualmente disponible)
if not _LANGCHAIN_AVAILABLE:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langchain_community.utilities import WikipediaAPIWrapper  # type: ignore[no-redef]
        _LANGCHAIN_AVAILABLE = True
        logger.debug("Usando langchain-community (fallback)")
    except ImportError:
        logger.warning(
            "OpenWikiTool deshabilitado. Para activarlo: "
            "pip install langchain-community wikipedia"
        )


class OpenWikiTool:
    """
    Herramienta de busqueda en Wikipedia para LegoGPT.

    Integra langchain-ai/openwiki mediante WikipediaAPIWrapper,
    permitiendo a agentes LLM consultar Wikipedia para enriquecer
    el contexto de generacion LEGO.

    Attributes:
        name: Identificador de la tool para agentes LangChain.
        description: Descripcion legible para el LLM.
        lang: Codigo ISO del idioma de Wikipedia (por defecto "es").
        top_k_results: Numero maximo de resultados a recuperar.
        doc_content_chars_max: Limite de caracteres del contenido.

    Example:
        tool = OpenWikiTool(lang="es", top_k_results=2)
        resultado = tool.search("LEGO Technic historia")
    """

    name: str = "openwiki"
    description: str = (
        "Busca informacion en Wikipedia sobre sets, piezas y tematicas LEGO. "
        "Util para enriquecer prompts con contexto enciclopedico. "
        "Input: cadena de texto con el termino a buscar. "
        "Output: resumen en texto plano extraido de Wikipedia."
    )

    def __init__(
        self,
        lang: str = "es",
        top_k_results: int = 3,
        doc_content_chars_max: int = 2000,
    ) -> None:
        """
        Inicializa OpenWikiTool.

        Args:
            lang: Codigo ISO del idioma de Wikipedia (ej: "es", "en", "de").
            top_k_results: Maximo de articulos a recuperar por consulta.
            doc_content_chars_max: Limite de caracteres del contenido devuelto.
        """
        self.lang = lang
        self.top_k_results = top_k_results
        self.doc_content_chars_max = doc_content_chars_max

        if _LANGCHAIN_AVAILABLE:
            self._wrapper = WikipediaAPIWrapper(
                lang=lang,
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
            )
        else:
            self._wrapper = None

    def search(self, query: str) -> str:
        """
        Busca en Wikipedia y devuelve el texto mas relevante.

        Args:
            query: Termino o frase a buscar en Wikipedia.

        Returns:
            Texto con el resumen encontrado en Wikipedia.

        Raises:
            ValueError: Si la query esta vacia o solo contiene espacios.
            RuntimeError: Si langchain-community no esta instalado.
        """
        if not query or not query.strip():
            raise ValueError("query no puede estar vacia")

        if self._wrapper is None:
            raise RuntimeError(
                "langchain-community no esta instalado. "
                "Ejecuta: pip install langchain-community wikipedia"
            )

        clean_query = query.strip()
        try:
            logger.info("OpenWikiTool.search: query=%r lang=%s", clean_query, self.lang)
            result: str = self._wrapper.run(clean_query)
            logger.debug("OpenWikiTool resultado: %d chars", len(result))
            return result
        except Exception as exc:
            logger.error("OpenWikiTool error al buscar %r: %s", clean_query, exc)
            return f"Resultado de Wikipedia no disponible para '{clean_query}': {exc}"

    def search_lego_context(self, query: str) -> str:
        """
        Busca en Wikipedia enriqueciendo la query con contexto LEGO.

        Anade el prefijo "LEGO" si no esta ya incluido en la query.

        Args:
            query: Termino a buscar (ej: "Technic", "Mindstorms", "Star Wars").

        Returns:
            Texto con informacion de Wikipedia relevante al dominio LEGO.
        """
        enriched = query.strip()
        if "lego" not in enriched.lower():
            enriched = f"LEGO {enriched}"
        return self.search(enriched)

    def as_langchain_tool(self):
        """
        Devuelve la tool como LangChain Tool nativo para uso en agentes.

        Returns:
            Instancia de langchain_core.tools.Tool lista para agentes LangChain.

        Raises:
            ImportError: Si langchain-core no esta instalado.

        Example:
            tool = OpenWikiTool()
            lc_tool = tool.as_langchain_tool()
        """
        try:
            from langchain_core.tools import Tool
        except ImportError as exc:
            raise ImportError(
                "langchain-core no esta instalado. "
                "Ejecuta: pip install langchain-core"
            ) from exc

        return Tool(
            name=self.name,
            description=self.description,
            func=self.search,
        )

    def __repr__(self) -> str:
        return (
            f"OpenWikiTool(lang={self.lang!r}, "
            f"top_k_results={self.top_k_results}, "
            f"doc_content_chars_max={self.doc_content_chars_max})"
        )


def get_lego_wiki_context(
    query: str,
    lang: str = "es",
    top_k_results: int = 2,
    doc_content_chars_max: int = 1500,
) -> str:
    """
    Funcion de conveniencia para obtener contexto de Wikipedia sobre LEGO.

    Crea una instancia temporal de OpenWikiTool y realiza la busqueda,
    anadiendo "LEGO" a la query si no esta incluido.

    Args:
        query: Termino a buscar (ej: "Technic", "Mindstorms EV3").
        lang: Codigo ISO del idioma de Wikipedia (por defecto "es").
        top_k_results: Maximo de articulos a recuperar.
        doc_content_chars_max: Limite de caracteres del resultado.

    Returns:
        Texto extraido de Wikipedia con informacion sobre el termino.

    Example:
        contexto = get_lego_wiki_context("LEGO Technic")
        print(contexto)
    """
    tool = OpenWikiTool(
        lang=lang,
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max,
    )
    return tool.search_lego_context(query)

"""
src/tools - Herramientas externas integradas en LegoGPT.

Herramientas disponibles:
- OpenWikiTool: Busqueda en Wikipedia via langchain-ai/openwiki
- get_lego_wiki_context: Funcion de conveniencia para contexto LEGO desde Wikipedia
"""
from src.tools.openwiki_tool import OpenWikiTool, get_lego_wiki_context

__all__ = ["OpenWikiTool", "get_lego_wiki_context"]

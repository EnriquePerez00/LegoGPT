"""
Tests para OpenWikiTool - integracion de langchain-ai/openwiki en LegoGPT.
Ejecutar: pytest tests/test_openwiki_tool.py -v
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_wikipedia_wrapper():
    wrapper = MagicMock()
    wrapper.run.return_value = (
        "LEGO es una empresa danesa de juguetes fundada en 1932. "
        "Los sets de LEGO Technic contienen engranajes y conectores."
    )
    return wrapper


@pytest.fixture
def openwiki_tool(mock_wikipedia_wrapper):
    from src.tools.openwiki_tool import OpenWikiTool
    with patch("src.tools.openwiki_tool.WikipediaAPIWrapper", return_value=mock_wikipedia_wrapper):
        tool = OpenWikiTool(lang="es", top_k_results=2, doc_content_chars_max=500)
    tool._wrapper = mock_wikipedia_wrapper
    return tool


class TestOpenWikiToolInit:
    def test_default_instantiation(self):
        from src.tools.openwiki_tool import OpenWikiTool
        with patch("src.tools.openwiki_tool.WikipediaAPIWrapper"):
            tool = OpenWikiTool()
            assert tool.lang == "es"
            assert tool.top_k_results == 3
            assert tool.doc_content_chars_max == 2000

    def test_custom_instantiation(self):
        from src.tools.openwiki_tool import OpenWikiTool
        with patch("src.tools.openwiki_tool.WikipediaAPIWrapper"):
            tool = OpenWikiTool(lang="en", top_k_results=5, doc_content_chars_max=1000)
            assert tool.lang == "en"
            assert tool.top_k_results == 5
            assert tool.doc_content_chars_max == 1000

    def test_tool_has_name_and_description(self):
        from src.tools.openwiki_tool import OpenWikiTool
        with patch("src.tools.openwiki_tool.WikipediaAPIWrapper"):
            tool = OpenWikiTool()
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert len(tool.name) > 0
            assert len(tool.description) > 0


class TestOpenWikiToolSearch:
    def test_search_returns_string(self, openwiki_tool):
        result = openwiki_tool.search("LEGO")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_search_calls_wrapper(self, openwiki_tool, mock_wikipedia_wrapper):
        openwiki_tool.search("LEGO Technic")
        mock_wikipedia_wrapper.run.assert_called_once_with("LEGO Technic")

    def test_search_lego_context_adds_prefix(self, openwiki_tool, mock_wikipedia_wrapper):
        openwiki_tool.search_lego_context("Technic")
        call_args = mock_wikipedia_wrapper.run.call_args[0][0]
        assert "LEGO" in call_args

    def test_search_lego_context_no_duplicate(self, openwiki_tool, mock_wikipedia_wrapper):
        openwiki_tool.search_lego_context("LEGO Technic")
        call_args = mock_wikipedia_wrapper.run.call_args[0][0]
        assert call_args.lower().count("lego") == 1

    def test_search_empty_query_raises(self, openwiki_tool):
        with pytest.raises(ValueError, match="query no puede estar vacia"):
            openwiki_tool.search("")

    def test_search_whitespace_query_raises(self, openwiki_tool):
        with pytest.raises(ValueError, match="query no puede estar vacia"):
            openwiki_tool.search("   ")

    def test_search_returns_fallback_on_network_error(self, openwiki_tool, mock_wikipedia_wrapper):
        mock_wikipedia_wrapper.run.side_effect = Exception("Network error")
        result = openwiki_tool.search("LEGO")
        assert "no disponible" in result.lower() or "error" in result.lower()

    def test_search_strips_whitespace(self, openwiki_tool, mock_wikipedia_wrapper):
        openwiki_tool.search("  LEGO  ")
        mock_wikipedia_wrapper.run.assert_called_once_with("LEGO")


class TestOpenWikiAsLangChainTool:
    def test_as_langchain_tool_callable(self, openwiki_tool):
        try:
            lc_tool = openwiki_tool.as_langchain_tool()
            assert callable(lc_tool.func) or hasattr(lc_tool, "run")
        except ImportError:
            pytest.skip("langchain-core no instalado")

    def test_as_langchain_tool_name(self, openwiki_tool):
        try:
            lc_tool = openwiki_tool.as_langchain_tool()
            assert lc_tool.name == openwiki_tool.name
        except ImportError:
            pytest.skip("langchain-core no instalado")


class TestGetLegoWikiContext:
    def test_returns_string(self):
        from src.tools.openwiki_tool import get_lego_wiki_context
        with patch("src.tools.openwiki_tool.WikipediaAPIWrapper") as MockWrapper:
            MockWrapper.return_value.run.return_value = "LEGO es una empresa danesa."
            result = get_lego_wiki_context("LEGO")
            assert isinstance(result, str)
            assert len(result) > 0

    def test_uses_spanish_by_default(self):
        from src.tools.openwiki_tool import get_lego_wiki_context
        with patch("src.tools.openwiki_tool.WikipediaAPIWrapper") as MockWrapper:
            MockWrapper.return_value.run.return_value = "contenido"
            get_lego_wiki_context("LEGO")
            call_kwargs = MockWrapper.call_args[1]
            assert call_kwargs.get("lang") == "es"

    def test_accepts_custom_lang(self):
        from src.tools.openwiki_tool import get_lego_wiki_context
        with patch("src.tools.openwiki_tool.WikipediaAPIWrapper") as MockWrapper:
            MockWrapper.return_value.run.return_value = "content"
            get_lego_wiki_context("LEGO", lang="en")
            call_kwargs = MockWrapper.call_args[1]
            assert call_kwargs.get("lang") == "en"

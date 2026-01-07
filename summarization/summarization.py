"""PDF summarizer with Rich output formatting.

Usage:
    python summarization.py --pdf path/to/file.pdf --format panel

Improvements:
- Uses Rich for better readable, wrapped and styled output (Panel/Markdown).
- Handles different response shapes from summarize chain.
- Adds a small CLI and status spinners.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from langchain_ollama import ChatOllama
# langchain's project layout changed in some installs; try both locations
try:
    from langchain.chains.summarize import load_summarize_chain  # type: ignore
except Exception:
    try:
        from langchain_classic.chains.summarize import load_summarize_chain  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time fallback
        raise ImportError(
            "Could not import load_summarize_chain from 'langchain.chains' or 'langchain_classic.chains'. "
            "Install the appropriate package or adjust your PYTHONPATH."
        ) from exc

from langchain_community.document_loaders import PyPDFLoader
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule

console = Console()


def _extract_text_from_chain_response(resp: Any) -> str:
    """Try to extract a readable summary string from various response shapes."""
    # Common shapes: string, dict with keys, objects with attributes, lists
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for key in ("output_text", "text", "summary", "result", "output"):
            if key in resp and isinstance(resp[key], str):
                return resp[key]
        # Fallback: join values that are strings
        pieces = [str(v) for v in resp.values() if isinstance(v, (str,))]
        if pieces:
            return "\n\n".join(pieces)
        return str(resp)
    if isinstance(resp, (list, tuple)):
        return "\n\n".join(_extract_text_from_chain_response(r) for r in resp)
    # Objects with attributes
    for attr in ("output_text", "text", "summary", "result"):
        if hasattr(resp, attr):
            val = getattr(resp, attr)
            if isinstance(val, str):
                return val
    # Fallback to string representation
    return str(resp)


def format_summary(summary_text: str, style: str = "panel", title: str | None = "Summary"):
    """Return a Rich renderable for the summary.

    style: 'panel' (default) wraps summary in a Panel, 'markdown' renders as Markdown,
    'plain' prints a wrapped Text object.
    """
    summary_text = summary_text.strip()
    if not summary_text:
        return Panel(Text("(no summary produced)", style="dim"), title=title)

    if style == "markdown":
        # Let Markdown handle line breaks and basic structure
        return Panel(Markdown(summary_text), title=title)

    if style == "plain":
        return Panel(Text(summary_text, overflow="fold"), title=title)

    # Default: panel with wrapped Text
    text = Text.from_markup(summary_text, justify="left")
    text.wrap = True
    return Panel(text, title=title)


def summarize_pdf(
    pdf_path: str,
    custom_prompt: str = "",
    model: str = "mistral-large-3:675b-cloud",
    temperature: float = 0.1,
    chain_type: str = "map_reduce",
) -> str:
    """Load a PDF and run the LangChain summarize chain.

    Returns the raw summary text (not a Rich renderable).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(file_path=pdf_path)
    with console.status("Loading PDF and splitting into documents..."):
        docs = loader.load_and_split()

    llm = ChatOllama(model=model, temperature=temperature)

    # Build the chain; pass prompt only if provided
    kwargs = {"chain_type": chain_type}
    if custom_prompt:
        kwargs["prompt"] = custom_prompt

    chain = load_summarize_chain(llm, **kwargs)

    with console.status("Generating summary..."):
        resp = chain.invoke(docs)

    return _extract_text_from_chain_response(resp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PDF and display it with Rich formatting.")
    parser.add_argument("--pdf", "-p", help="Path to the PDF file")
    parser.add_argument("--prompt", help="Custom prompt to use for summarization", default="")
    parser.add_argument(
        "--format",
        "-f",
        choices=("panel", "markdown", "plain"),
        default="panel",
        help="Output format to render the summary",
    )
    parser.add_argument("--model", default="mistral-large-3:675b-cloud", help="Model to use")
    parser.add_argument("--temp", type=float, default=0.1, help="LLM temperature")

    args = parser.parse_args()

    if not args.pdf:
        console.print(Rule("PDF Summarizer"))
        console.print("=" * 10)
        console.print("""

:'######::'##::::'##:'##::::'##:'##::::'##::::'###::::'########::'####:'########:'########:'########::
'##... ##: ##:::: ##: ###::'###: ###::'###:::'## ##::: ##.... ##:. ##::..... ##:: ##.....:: ##.... ##:
 ##:::..:: ##:::: ##: ####'####: ####'####::'##:. ##:: ##:::: ##:: ##:::::: ##::: ##::::::: ##:::: ##:
. ######:: ##:::: ##: ## ### ##: ## ### ##:'##:::. ##: ########::: ##::::: ##:::: ######::: ########::
:..... ##: ##:::: ##: ##. #: ##: ##. #: ##: #########: ##.. ##:::: ##:::: ##::::: ##...:::: ##.. ##:::
'##::: ##: ##:::: ##: ##:.:: ##: ##:.:: ##: ##.... ##: ##::. ##::: ##::: ##:::::: ##::::::: ##::. ##::
. ######::. #######:: ##:::: ##: ##:::: ##: ##:::: ##: ##:::. ##:'####: ########: ########: ##:::. ##:
:......::::.......:::..:::::..::..:::::..::..:::::..::..:::::..::....::........::........::..:::::..::

""")
        console.print("="*10)
        console.print("Enter a PDF path (or press Enter to quit):")
        while True:
            try:
                pdf_path = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                return
            if not pdf_path:
                console.print("Goodbye.")
                return

            try:
                summary = summarize_pdf(pdf_path, custom_prompt=args.prompt, model=args.model, temperature=args.temp)
                console.print(format_summary(summary, style=args.format))
            except Exception as exc:  # noqa: BLE001
                console.print(Panel(Text(str(exc), style="bold red"), title="Error", style="red"))
    else:
        try:
            summary = summarize_pdf(args.pdf, custom_prompt=args.prompt, model=args.model, temperature=args.temp)
            console.print(Rule("Summary"))
            console.print(format_summary(summary, style=args.format))
        except Exception as exc:  # noqa: BLE001
            console.print(Panel(Text(str(exc), style="bold red"), title="Error", style="red"))


if __name__ == "__main__":
    main()

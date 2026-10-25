"""
Gradio chat UI over a :class:`~local_agentic_rag.pipeline.RAGPipeline`.

    from local_agentic_rag.app import launch_chat_ui
    launch_chat_ui(pipe)            # in Colab prints a public *.gradio.live link

Layout: left column = PDF upload, ingest status and document selector;
middle = chat; right = citations and agent trace for the last answer.

Gradio is an optional dependency, so it is imported inside the functions.
Every event handler catches its own exceptions and shows a friendly message,
because an exception raised into Gradio leaves the user with a generic
red "Error" box and a stuck spinner.
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_agentic_rag.colab import render_citations_html, render_trace_html

if TYPE_CHECKING:  # pragma: no cover
    import gradio as gr

    from local_agentic_rag.pipeline import RAGPipeline
    from local_agentic_rag.schema import Answer, IngestReport

logger = logging.getLogger(__name__)

ALL_DOCS = "All documents"
_EMPTY_SIDE = "<i>Ask a question to see its sources and the agent trace here.</i>"


# --------------------------------------------------------------------------- pure helpers


def history_pairs(pairs: list[list[str]] | list[tuple[str, str]] | None) -> list[tuple[str, str]]:
    """Normalise stored (user, assistant) turns into the tuples ``pipe.ask`` expects."""
    return [(str(u), str(a)) for u, a in (pairs or [])]


def doc_choices(pipe: RAGPipeline) -> list[str]:
    return [ALL_DOCS, *sorted(pipe.indexes)]


def selected_doc(choice: str | None) -> str | None:
    return None if not choice or choice == ALL_DOCS else choice


def format_ingest_status(reports: list[IngestReport], errors: list[str]) -> str:
    lines = []
    for r in reports:
        line = (
            f"**{r.doc_id}**: {r.num_pages} pages, {r.num_chunks} chunks "
            f"({r.num_table_chunks} tables), {r.seconds:.1f}s"
        )
        if r.cached:
            line += " (cached)"
        if r.ocr_pages:
            line += f", OCR pages {r.ocr_pages}"
        lines.append(f"- {line}")
        lines.extend(f"  - warning: {w}" for w in r.warnings)
    lines.extend(f"- {e}" for e in errors)
    return "\n".join(lines) or "No files uploaded."


def side_panel_html(answer: Answer) -> str:
    if answer.error:
        status = "error"
    elif answer.refused:
        status = "not found in document"
    elif answer.grounded is None:
        status = "not verified"
    else:
        status = "verified" if answer.grounded else "partially unverified"
    queries = ", ".join(answer.search_queries) if answer.search_queries else "-"
    return (
        f"<div><b>Status:</b> {html.escape(status)} &middot; {answer.latency_sec:.1f}s &middot; "
        f"{answer.llm_calls} LLM calls</div>"
        f"<div style='margin-top:4px'><b>Search queries:</b> {html.escape(queries)}</div>"
        f"<h4 style='margin:10px 0 4px'>Citations</h4>{render_citations_html(answer.citations, 'ui')}"
        f"{render_trace_html(answer.trace, open_=True)}"
    )


def chat_reply_text(answer: Answer) -> str:
    """What the chat bubble shows: the answer plus a compact page footer."""
    text = answer.answer
    if answer.error:
        text += f"\n\n_(problem: {answer.error})_"
    elif answer.cited_pages:
        text += "\n\n_Pages: " + ", ".join(map(str, answer.cited_pages)) + "_"
    if answer.grounded is False:
        text += "\n\n_Some statements could not be verified against the document._"
    return text


# --------------------------------------------------------------------------- UI


def _chatbot(gr: Any) -> Any:
    try:
        return gr.Chatbot(type="messages", label="Chat", height=520)
    except TypeError:  # Gradio 6 removed ``type``; messages are the only format
        return gr.Chatbot(label="Chat", height=520)


def build_chat_ui(pipe: RAGPipeline) -> gr.Blocks:
    import gradio as gr

    def on_ingest(files: list[Any] | None, current: str | None):
        reports, errors = [], []
        for f in files or []:
            path = Path(getattr(f, "name", f))
            if path.suffix.lower() != ".pdf":
                errors.append(f"{path.name}: skipped (not a PDF)")
                continue
            try:
                reports.append(pipe.ingest(path))
            except Exception as exc:  # noqa: BLE001 - show, never raise into Gradio
                logger.exception("ingest failed for %s", path)
                errors.append(f"{path.name}: could not be processed ({type(exc).__name__}: {exc})")
        choices = doc_choices(pipe)
        value = reports[-1].doc_id if reports else (current if current in choices else ALL_DOCS)
        return format_ingest_status(reports, errors), gr.update(choices=choices, value=value)

    def on_ask(question: str, messages: list[dict], pairs: list, doc_choice: str | None):
        messages = list(messages or [])
        pairs = list(pairs or [])
        question = (question or "").strip()
        if not question:
            return "", messages, pairs, gr.update()
        messages.append({"role": "user", "content": question})
        if not pipe.indexes:
            messages.append(
                {"role": "assistant", "content": "Please upload and process a PDF first."}
            )
            return "", messages, pairs, gr.update()
        try:
            answer = pipe.ask(question, selected_doc(doc_choice), history=history_pairs(pairs))
        except Exception as exc:  # noqa: BLE001
            logger.exception("ask failed")
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Sorry, something went wrong: {type(exc).__name__}: {exc}",
                }
            )
            return "", messages, pairs, gr.update()
        messages.append({"role": "assistant", "content": chat_reply_text(answer)})
        if not answer.error:
            pairs.append([question, answer.answer])
        return "", messages, pairs, side_panel_html(answer)

    def on_clear():
        return [], [], _EMPTY_SIDE

    with gr.Blocks(title="Local Agentic RAG") as demo:
        gr.Markdown(
            "## Local Agentic RAG\nUpload PDFs, then ask questions. Answers cite pages; "
            "the right panel shows the sources and every agent step."
        )
        pairs_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                files = gr.File(
                    label="PDFs", file_count="multiple", file_types=[".pdf"], type="filepath"
                )
                ingest_btn = gr.Button("Process PDFs", variant="primary")
                status = gr.Markdown(
                    format_ingest_status([], [])
                    if not pipe.indexes
                    else f"{len(pipe.indexes)} document(s) loaded."
                )
                doc = gr.Dropdown(choices=doc_choices(pipe), value=ALL_DOCS, label="Search in")
            with gr.Column(scale=3):
                chat = _chatbot(gr)
                question = gr.Textbox(placeholder="Ask about the document...", show_label=False)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear chat")
            with gr.Column(scale=2, min_width=280):
                gr.Markdown("### Sources & trace")
                side = gr.HTML(_EMPTY_SIDE)

        ingest_btn.click(on_ingest, [files, doc], [status, doc])
        ask_io = ([question, chat, pairs_state, doc], [question, chat, pairs_state, side])
        send.click(on_ask, *ask_io)
        question.submit(on_ask, *ask_io)
        clear.click(on_clear, None, [chat, pairs_state, side])
    return demo


def launch_chat_ui(pipe: RAGPipeline, share: bool = True, **launch_kwargs: Any) -> gr.Blocks:
    """Build and launch the UI. In Colab ``share=True`` prints a public link."""
    demo = build_chat_ui(pipe)
    demo.queue(default_concurrency_limit=1)  # one GPU, one question at a time
    demo.launch(share=share, **launch_kwargs)
    return demo

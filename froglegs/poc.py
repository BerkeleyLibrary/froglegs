import re
import textwrap

import rich
import typer

from pathlib import Path
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from pdffile import PDFFile, add_directory

PROMPT = PromptTemplate.from_template(
    """
    You are a reference librarian who helps researchers answer
    questions about information in oral history interviews. Use the
    following pieces of retrieved context to answer the question. If
    you don't know the answer, just say that you don't know. Use three
    sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
)

USER_PROMPT = ":speaking_head: [magenta] What's your question?[/magenta]"

app = typer.Typer()


def _enumerate_filepaths(filepaths: List[str], recurse: bool) -> List[PDFFile]:
    """
    Take a list of filepaths and transform it into a list of File objects,
    optionally recursing into each of them.

    Args:
        filepaths (List[str]): a list of files or paths for loading
        recurse (bool): whether to recurse into each given filepath

    Returns:
        List[PDFFile]: A list of File objects representing the files extracted
        from all filepaths.

    Raises:
        FileNotFoundError: If a filepath does not exist.
        IsADirectoryError: If recurse is False and a filepath points to a
        directory instead of a file.
    """
    if not recurse:
        return [PDFFile(filepath=filepath) for filepath in filepaths]

    files = []
    for fp in filepaths:
        files.extend(
            add_directory(fp) if Path(fp).is_dir() else [PDFFile(filepath=fp)]
        )
    return files


@app.command()
def main(
    filepaths: Optional[List[str]] = typer.Argument(
        default=None,
        help="A list of filepaths to load.",
    ),
    display_prompt: Optional[bool] = typer.Option(
        default=False,
        help="Print the prompt used before returning the answer."
    ),
    recurse: Optional[bool] = typer.Option(
        default=False,
        help="Enable recursion into filepaths.",
    ),
    embedding_model: Optional[str] = typer.Option(
        default="nomic-embed-text",
        help="Embedding model to use to generate vector embeddings.",
    ),
    generator_model: Optional[str] = typer.Option(
        default="phi4-mini", help="LLM to use for response generation."
    ),
):
    """
    Proof of concept end to end RAG, running entirely locally.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)
    llm = OllamaLLM(model=generator_model)

    rich.print(
        ":frog: [green]frog[/green][bold yellow]legs[/bold yellow]",
        f"| using [yellow]{embedding_model}[/yellow]",
        f"& [magenta]{generator_model}[/magenta]",
    )

    if filepaths is None:
        filepaths = []

    files = _enumerate_filepaths(filepaths=filepaths, recurse=recurse)

    if files:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            total = 0
            total_chunks = 0
            lt = progress.add_task(
                description=":person_lifting_weights: Loading files...",
                total=len(files),
            )
            for file in files:
                fn = file.filepath
                progress.update(
                    lt,
                    description=f":file_folder: Loading {fn}...",
                    advance=0.3,
                )
                loader = PyPDFLoader(file.filepath)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200, add_start_index=True
                )
                all_splits = text_splitter.split_documents(docs)
                chunks = len(all_splits)
                total_chunks += chunks
                progress.update(
                    lt,
                    description=f":knife: Got {chunks} chunks for {fn}.",
                    advance=0.1,
                )

                progress.update(
                    lt,
                    description=f":abacus: Creating embeddings for {fn}...",
                    advance=0.3,
                )
                if total == 0:
                    vector_1 = embeddings.embed_query(
                        all_splits[0].page_content
                    )
                    v = len(vector_1)
                    rich.print(
                        ":abacus: Generating vectors of length",
                        f"{v} using [yellow]{embedding_model}[/yellow].",
                    )

                progress.update(
                    lt,
                    description=f":inbox_tray: Storing vectors for {fn}...",
                    advance=0.3,
                )
                vector_store.add_documents(documents=all_splits)
                total += 1

        rich.print(
            f":jigsaw: Parsed {total} files into {total_chunks}",
            "chunks.\n:information_desk_person:",
            f"[magenta]{generator_model}[/magenta] is ready.",
        )

        if display_prompt:
            display = re.sub(r" {4}", "   ", PROMPT.template.strip())
            rich.print(
                ":mega: [bold blue]Prompt:[/bold blue]",
                f"[blue]{display}[/blue]",
                )
        query = Prompt.ask(USER_PROMPT)
        query_vector = embeddings.embed_query(query)
        retrieved_docs = vector_store.similarity_search_by_vector(query_vector)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = PROMPT.invoke({"question": query, "context": context})
        rich.print(
            ":information_desk_person: [bold cyan]Result:[/bold cyan]",
            f"[cyan]{textwrap.fill(
                llm.invoke(prompt),
                width=67,
                subsequent_indent='   '
            )}[/cyan]",
        )

    else:
        rich.print(f":stop_sign: [red]Nothing to do! {0} files given.[/red]")


if __name__ == "__main__":
    app()

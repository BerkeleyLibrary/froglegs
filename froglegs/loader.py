import rich
import typer

from pathlib import Path
from typing import List, Optional

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.progress import Progress, SpinnerColumn, TextColumn

from pdffile import PDFFile

app = typer.Typer()

def _enumerate_filepaths(filepaths: List[str], recurse: bool) -> List[PDFFile]:
    """
    Take a list of filepaths and transform it into a list of File objects, optionally recursing into each of them.

    Args:
        filepaths (List[str]): a list of files or paths for loading
        recurse (bool): whether to recurse into each given filepath

    Returns:
        List[PDFFile]: A list of File objects representing the files extracted from all filepaths.
    
    Raises:
        FileNotFoundError: If a filepath does not exist.
        IsADirectoryError: If recurse is False and a filepath points to a directory instead of a file.
    """
    if not recurse:
        return [PDFFile(filepath=filepath) for filepath in filepaths]

    files = []
    for fp in filepaths:
        files.extend(_enumerate_filepaths(fp, recurse) if Path(fp).is_dir() else [PDFFile(filepath=fp)])
    return files

@app.command()
def main(
    filepaths: Optional[List[str]] = typer.Argument(
        default=None,
        help="A list of filepaths to load.",
    ),
    recurse: Optional[bool] = typer.Option(
        default=False,
        help="Enable recursion into filepaths. [NOT YET IMPLEMENTED]",
    ),
    embedding_model: Optional[str] = typer.Option(
        default="nomic-embed-text",
        help="Embedding model to use to generate vector embeddings."
    )
):
    """
    Process a set of PDF files.
    """
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = InMemoryVectorStore(embeddings)

    print("🐸 froglegs")

    if filepaths is None:
        filepaths = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Loading %s..." % filepaths[0], total=None)
        loader = PyPDFLoader(filepaths[0]) # hack :/
        docs = loader.load()

        progress.add_task(description="Chunking %s..." % filepaths[0], total=None)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        rich.print(f"Returned {len(all_splits)} chunks for {filepaths[0]}.")

        progress.add_task(description="Creating embeddings ...", total=None)
        vector_1 = embeddings.embed_query(all_splits[0].page_content)
        rich.print(f"Generating vectors of length {len(vector_1)}.")

        progress.add_task(description="Storing vectors ...", total=None)
        ids = vector_store.add_documents(documents=all_splits)

    query = typer.prompt("What's your question?")
    query_vector = embeddings.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_vector)

    for result in results:
        rich.print(result)

if __name__ == "__main__":
    app()
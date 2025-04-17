import rich
import typer

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    print("🐸 froglegs")

    if filepaths is None:
        filepaths = []

    embeddings = OllamaEmbeddings(model=embedding_model)

    loader = PyPDFLoader(filepaths[0]) # hack :/
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    rich.print(len(all_splits))
    rich.print(all_splits[3])

if __name__ == "__main__":
    app()
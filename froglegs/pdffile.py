import os

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PDFFile(BaseModel):
    """
    Represents a PDF file with properties, metadata, etc. for processing

    Attributes:
        filepath (str): The path to the file
        tind_id (Optional[str]): The ID for the associated TIND object
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    filepath: str = Field(..., exclude=True)
    tind_id: Optional[str] = Field(default=None)

    @staticmethod
    def _validate_filepath(path):
        """
        Validates if the given filepath exists and is a file.

        Args:
            path (str): The filepath to be validated.

        Raises:
            FileNotFoundError: If the filepath does not exist.
            IsADirectoryError: If the filepath points to a directory instead
                               of a file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Filepath {path} does not exist.")
        elif not os.path.isfile(path):
            raise IsADirectoryError(f"Filepath {path} is not a file.")


def add_directory(directory: str):
    """
    Recursively adds all PDF files in the specified directory to a list
    of PDFFile objects.
    """

    files = []

    for file in Path(directory).rglob("*.pdf"):
        if not file.is_file():
            continue

        files.append(PDFFile(filepath=str(file)))

    return files

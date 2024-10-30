from fastapi import UploadFile, HTTPException
from typing import Set

class FileValidator:
    def __init__(self, allowed_extensions: Set[str]):
        self.allowed_extensions = allowed_extensions

    def validate(self, file: UploadFile) -> None:
        """Validates file before processing"""
        if not self._is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file type",
                    "allowed_extensions": list(self.allowed_extensions)
                }
            )

    def _is_allowed_file(self, filename: str) -> bool:
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions 
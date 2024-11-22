class VectorDBClientError(Exception):
    """Base exception for VectoDBClient"""
    pass

class VectorDBClientConnectionError(VectorDBClientError):
    """Raised when the client fails to connect to the server"""
    pass

class VectorDBClientRequestError(VectorDBClientError):
    """Raised when a request to the server fails"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")

class VectorDBClientValidationError(VectorDBClientError):
    """Raised when data validation fails"""
    pass
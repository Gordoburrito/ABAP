"""Custom exceptions for the incomplete fitment data processing pipeline"""


class DataProcessingError(Exception):
    """Base exception for data processing errors"""
    pass


class DataValidationError(DataProcessingError):
    """Exception raised when data validation fails"""
    pass


class AIExtractionError(DataProcessingError):
    """Exception raised when AI extraction fails"""
    pass


class GoldenMasterValidationError(DataProcessingError):
    """Exception raised when golden master validation fails"""
    pass


class TokenEstimationError(DataProcessingError):
    """Exception raised when token estimation fails"""
    pass


class ShopifyFormatError(DataProcessingError):
    """Exception raised when Shopify format generation fails"""
    pass


class BatchProcessingError(DataProcessingError):
    """Exception raised when batch processing fails"""
    pass
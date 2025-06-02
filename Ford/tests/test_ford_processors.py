"""
Tests for Ford data processing.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from single_processor import FordSingleProcessor
from batch_processor import FordBatchProcessor


class TestFordSingleProcessor(unittest.TestCase):
    """Test cases for Ford single processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FordSingleProcessor()
    
    def test_process_item(self):
        """Test basic item processing."""
        test_item = {
            "id": "test_001",
            "title": "Test Item",
            "description": "This is a test item"
        }
        
        result = self.processor.process_item(test_item)
        
        # Check that basic processing occurred
        self.assertIn("data_source", result)
        self.assertEqual(result["data_source"], "Ford")
        self.assertIn("processing_timestamp", result)
        self.assertIn("processing_completed", result)


class TestFordBatchProcessor(unittest.TestCase):
    """Test cases for Ford batch processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FordBatchProcessor()
    
    def test_process_batch(self):
        """Test basic batch processing."""
        test_batch = [
            {"id": "test_001", "title": "Test Item 1"},
            {"id": "test_002", "title": "Test Item 2"}
        ]
        
        results = self.processor.process_batch(test_batch)
        
        # Check that all items were processed
        self.assertEqual(len(results), 2)
        
        # Check that each item has required fields
        for result in results:
            self.assertIn("data_source", result)
            self.assertEqual(result["data_source"], "Ford")


if __name__ == "__main__":
    unittest.main()

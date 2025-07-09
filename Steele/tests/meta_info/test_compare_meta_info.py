import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import io
import sys
import os

# Add the project source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from project.src.ABAP.meta_info.compare_meta_info import (
    load_data, find_unique_collections, get_sample_from_each_collection,
    find_matching_skus, compare_rows_by_sku, compare_all_dataframes
)


class TestCompareMetaInfo(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_data1 = """SKU,Collection,Title,Description
123,Winter,Product 1,Description 1
456,Summer,Product 2,Description 2
789,Spring,Product 3,Description 3"""

        self.sample_data2 = """SKU,Collection,Title,Description
123,Winter,Product 1 Updated,New Description 1
456,Summer,Product 2 V2,New Description 2
999,Fall,Product 4,Description 4"""

        # Create sample dataframes
        self.df1 = pd.read_csv(io.StringIO(self.sample_data1))
        self.df2 = pd.read_csv(io.StringIO(self.sample_data2))

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        # Mock pandas read_csv to return our sample dataframes
        mock_read_csv.side_effect = [
            self.df1, self.df2, self.df1,  # Main dataframes
            self.df2, self.df1, self.df2   # Mini dataframes
        ]
        
        # Test loading data
        result = load_data()
        self.assertEqual(len(result), 6)
        self.assertTrue('df1' in result)
        self.assertTrue('df2' in result)
        self.assertTrue('df3' in result)
        self.assertTrue('df1_mini' in result)
        self.assertTrue('df2_mini' in result)
        self.assertTrue('df3_mini' in result)
        
        # Verify read_csv was called with correct paths
        expected_calls = [
            os.path.join("project/src/ABAP/meta_info/data/processed/meta title and meta description/", "1-4o.csv"),
            os.path.join("project/src/ABAP/meta_info/data/processed/meta title and meta description/", "2-4o.csv"),
            os.path.join("project/src/ABAP/meta_info/data/processed/meta title and meta description/", "3-4o.csv"),
            os.path.join("project/src/ABAP/meta_info/data/processed/meta title and meta description/", "1-4o-mini.csv"),
            os.path.join("project/src/ABAP/meta_info/data/processed/meta title and meta description/", "2-4o-mini.csv"),
            os.path.join("project/src/ABAP/meta_info/data/processed/meta title and meta description/", "3-4o-mini.csv")
        ]
        
        for i, call in enumerate(mock_read_csv.call_args_list):
            self.assertEqual(call[0][0], expected_calls[i])

    def test_find_unique_collections(self):
        # Test finding unique collections using the actual function
        collections = find_unique_collections(self.df1)
        self.assertEqual(len(collections), 3)
        self.assertTrue('Winter' in collections)
        self.assertTrue('Summer' in collections)
        self.assertTrue('Spring' in collections)

    def test_get_sample_from_each_collection(self):
        # Test getting sample rows from each collection
        samples = get_sample_from_each_collection(self.df1)
        self.assertEqual(len(samples), 3)  # One row per collection
        
        # Verify we have one row from each collection
        collections = samples['Collection'].tolist()
        self.assertTrue('Winter' in collections)
        self.assertTrue('Summer' in collections)
        self.assertTrue('Spring' in collections)

    def test_find_matching_skus(self):
        # Test finding matching SKUs between dataframes using the actual function
        common_skus = find_matching_skus(self.df1, self.df2)
        self.assertEqual(len(common_skus), 2)
        self.assertTrue('123' in common_skus)
        self.assertTrue('456' in common_skus)

    def test_compare_rows_by_sku(self):
        # Test comparing rows with the same SKU using the actual function
        sku = '123'

        print(self.df1)
        print(self.df2)
        differences = compare_rows_by_sku(self.df1, self.df2, sku)
        

        print(differences)
        self.assertIsNotNone(differences)
        self.assertTrue('Title' in differences)
        self.assertTrue('Description' in differences)
        self.assertEqual(differences['Title']['df1'], 'Product 1')
        self.assertEqual(differences['Title']['df2'], 'Product 1 Updated')
        
        # Test with a non-existent SKU
        nonexistent_sku = '999'
        differences = compare_rows_by_sku(self.df1, self.df2, nonexistent_sku)
        self.assertIsNone(differences)

    @patch('project.src.ABAP.meta_info.compare_meta_info.load_data')
    @patch('project.src.ABAP.meta_info.compare_meta_info.print')
    def test_compare_all_dataframes(self, mock_print, mock_load_data):
        # Create a mock data dictionary
        mock_data = {
            'df1_mini': self.df1,
            'df2_mini': self.df2,
        }
        mock_load_data.return_value = mock_data
        
        # Run the comparison function
        compare_all_dataframes()
        
        # Verify that load_data was called
        mock_load_data.assert_called_once()
        
        # Verify that print was called (at least once)
        self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main()

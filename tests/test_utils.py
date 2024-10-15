import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

from src.utils.utils import read_file_based_on_type, extract_text_from_pdf, read_text_file

class TestReadFileBasedOnType(unittest.TestCase):

    @patch('src.utils.utils.extract_text_from_pdf')
    def test_read_pdf_file(self, mock_extract_text_from_pdf):
        mock_extract_text_from_pdf.return_value = 'PDF content'
        file_path = 'path/to/file.pdf'
        result = read_file_based_on_type(file_path)
        self.assertEqual(result, 'PDF content')
        mock_extract_text_from_pdf.assert_called_once_with(file_path)
    
    @patch('src.utils.utils.read_text_file')
    def test_read_txt_file(self, mock_read_text_file):
        mock_read_text_file.return_value = 'Text content'
        file_path = 'path/to/file.txt'
        result = read_file_based_on_type(file_path)
        self.assertEqual(result, 'Text content')
        mock_read_text_file.assert_called_once_with(file_path)
    
    def test_read_unsupported_file_type(self):
        file_path = 'path/to/file.docx'
        with self.assertRaises(ValueError):
            read_file_based_on_type(file_path)
    
    def test_read_file_with_unknown_extension(self):
        file_path = 'path/to/file'
        with self.assertRaises(ValueError):
            read_file_based_on_type(file_path)
    
    def test_read_file_with_missing_extension(self):
        file_path = 'path/to/file.'
        with self.assertRaises(ValueError):
            read_file_based_on_type(file_path)

if __name__ == '__main__':
    unittest.main()
import unittest
import os
from corebehrt.functional.files_and_folders import split_path


class TestSplitPath(unittest.TestCase):
    def test_split_path(self):
        # Test with a simple path
        path = os.path.join("home", "user", "documents")
        expected = ["home", "user", "documents"]
        self.assertEqual(split_path(path), expected)

        # Test with a path with leading slash
        path = os.path.join(os.sep, "home", "user", "documents")
        expected = [os.sep, "home", "user", "documents"]
        self.assertEqual(split_path(path), expected)

        # Test with a path with trailing slash
        path = os.path.join("home", "user", "documents") + os.sep
        expected = ["home", "user", "documents"]
        self.assertEqual(split_path(path), expected)

        # Test with a path with multiple slashes
        path = os.path.join("home", "", "user", "documents")
        expected = ["home", "user", "documents"]
        self.assertEqual(split_path(path), expected)

        # Test with an empty path
        path = ""
        expected = []
        self.assertEqual(split_path(path), expected)


if __name__ == "__main__":
    unittest.main()

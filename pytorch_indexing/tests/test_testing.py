from unittest import TestCase

import pytorch_indexing
from pytorch_indexing import testing

class TestTesting(TestCase):
    def test_is_string(self):
        s = testing.hello()
        self.assertTrue(isinstance(s, str))

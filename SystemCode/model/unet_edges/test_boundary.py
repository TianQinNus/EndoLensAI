import unittest
import numpy as np
from predict import boundary


class BoundaryTests(unittest.TestCase):
    def test_empty_and_full(self):
        self.assertFalse(boundary(np.zeros((5, 5), bool)).any())
        edge = boundary(np.ones((5, 5), bool))
        self.assertEqual(int(edge.sum()), 16)
        self.assertFalse(edge[1:4, 1:4].any())

    def test_square_and_single_pixel(self):
        mask = np.zeros((7, 7), bool)
        mask[2:5, 2:5] = True
        edge = boundary(mask)
        self.assertEqual(int(edge.sum()), 8)
        self.assertFalse(edge[3, 3])
        self.assertFalse((edge & ~mask).any())
        self.assertTrue(boundary(np.ones((1, 1), bool))[0, 0])


if __name__ == '__main__':
    unittest.main()

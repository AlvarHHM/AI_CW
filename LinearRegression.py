import unittest


class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_function(self):
        self.assertEqual(vector_affinity([1, 2, 3], [1, 2, 3, 4, 5]), 0.6)
        self.assertEqual(vector_affinity([1, 2, 3, 4], [1, 2, 3, 5]), 0.75)
        self.assertEqual(vector_affinity([6, 6, 6, 6, 6, 6], [6]), 1.0 / 6.0)
        self.assertEqual(vector_affinity([None], [None]), 1.0)
        self.assertEqual(vector_affinity([None], []), 0.0)
        self.assertEqual(vector_affinity([None], [None, None]), 0.5)
        self.assertEqual(vector_affinity([], []), 1.0)


def vector_affinity(a, b):
    a, b = (a, b) if len(a) < len(b) else (b, a)
    count = 0.0
    for i in range(len(a)):
        if a[i] == b[i]:
            count += 1
    return count / len(b) if not len(b) == 0 else 1 if len(a) == 0 else 0


if __name__ == '__main__':
    unittest.main()
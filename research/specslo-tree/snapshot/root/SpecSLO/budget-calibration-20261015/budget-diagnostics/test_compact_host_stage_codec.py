import unittest
from compact_host_stage_codec import encode, decode_success


class CodecTest(unittest.TestCase):
    def rows(self, name='live-prefix-write-declaration', sequence=17):
        return [encode(rank, sequence, name, None) for rank in range(4)]

    def test_exact_success_values(self):
        for name in ('', 'mixed-target-completion', 'a'*64, '中文阶段', 'x\0y'):
            for sequence in (0, 17, 2**40):
                expected = [dict(rank=r, sequence=sequence, stage=name, error=None) for r in range(4)]
                self.assertEqual(decode_success(self.rows(name, sequence)), expected)

    def test_error_on_any_rank_requires_original_objects(self):
        for rank in range(4):
            for error in ('ValueError: failed', '', 'KeyboardInterrupt: '):
                rows = self.rows()
                rows[rank] = encode(rank, 17, 'live-prefix-write-declaration', error)
                self.assertIsNone(decode_success(rows))

    def test_long_or_nonstring_names_require_original_objects(self):
        for name in ('x'*65, '中'*22, 'bad\ud800', None, 3, ('stage',)):
            rows = self.rows(); rows[2] = encode(2, 17, name, None)
            self.assertIsNone(decode_success(rows))

    def test_cross_rank_disagreement_never_accepted(self):
        for rank in range(4):
            for altered in (encode(rank, 18, 'live-prefix-write-declaration', None),
                            encode(rank, 17, 'other-stage', None),
                            encode(4, 17, 'live-prefix-write-declaration', None)):
                rows = self.rows(); rows[rank] = altered
                self.assertIsNone(decode_success(rows))

    def test_rank_order_is_retained(self):
        rows = self.rows()[::-1]
        self.assertEqual([r['rank'] for r in decode_success(rows)], [3, 2, 1, 0])

    def test_malformed_payload_requires_original_objects(self):
        for index, value in ((2, 2), (3, 65), (4, 2**80), (4, 0)):
            rows = self.rows(); row = list(rows[1]); row[index] = value; rows[1] = row
            self.assertIsNone(decode_success(rows))
        rows = self.rows(); rows[1] = rows[1][:-1]
        self.assertIsNone(decode_success(rows))


if __name__ == '__main__':
    unittest.main()

import unittest

import decision as D


def box(x, y, w=100, h=100):
    return [x, y, w, h]


class DecisionTests(unittest.TestCase):
    def test_cross_class_cluster_keeps_both_scores(self):
        preds = [dict(image_id='a', category_id=D.RIPE, bbox=box(0, 0), score=0.9),
                 dict(image_id='a', category_id=D.UNRIPE, bbox=box(5, 5), score=0.3),
                 dict(image_id='a', category_id=D.UNRIPE, bbox=box(500, 500), score=0.8)]
        clusters = D.fruits(preds)
        self.assertEqual(len(clusters), 2)
        first = clusters[0]
        self.assertEqual((first['s_ripe'], first['s_unripe']), (0.9, 0.3))
        self.assertGreater(first['margin'], 0)
        self.assertLess(clusters[1]['margin'], 0)

    def test_matching_is_one_to_one(self):
        gt = [dict(bbox=box(0, 0), category_id=D.RIPE), dict(bbox=box(2, 2), category_id=D.RIPE)]
        clusters = [dict(bbox=box(1, 1))]
        self.assertEqual(len(D.match(gt, clusters)), 1)

    def test_thresholds_and_decisions(self):
        pairs = [(0.99, True), (0.95, True), (0.9, True), (0.7, False), (0.2, False), (0.05, False), (0.3, True)]
        low, high = D.choose_thresholds(pairs, target=1.0, floor_high=0.5, ceil_low=0.49)
        self.assertEqual(high, 0.71)
        self.assertEqual(low, 0.29)
        self.assertEqual(D.choose_thresholds([(0.99, True), (0.01, False)]), (0.1, 0.9))
        self.assertEqual(D.decide(0.95, low, high), 'Panen')
        self.assertEqual(D.decide(0.5, low, high), 'Cek manual')
        self.assertEqual(D.decide(0.1, low, high), 'Belum')

    def test_evaluate_counts_premature_and_missed(self):
        rows = [dict(clusters=[dict(margin=5.0, s_ripe=0.9, s_unripe=0.01)], gt=[dict(category_id=D.UNRIPE)], match={0: 0}),
                dict(clusters=[], gt=[dict(category_id=D.RIPE)], match={})]
        m = D.evaluate(rows, D.Platt(1, 0), 0.2, 0.8)
        self.assertEqual(m['premature'], dict(k=1, n=1))
        self.assertEqual(m['missed_ripe'], dict(k=1, n=1))
        self.assertEqual(m['confusion']['ripe->Tidak terdeteksi'], 1)
        om = D.evaluate(rows, None, 0, 1, 'omyon')
        self.assertEqual(om['confusion']['unripe->Panen'], 1)

    def test_tau_det_median_of_ties(self):
        rows = [dict(gt=[dict(bbox=box(0, 0), category_id=D.RIPE)], clusters=[dict(bbox=box(0, 0), confidence=0.97)])]
        self.assertEqual(D.choose_tau_det(rows), 0.5)

    def test_wilson(self):
        lo, hi = D.wilson(0, 12)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.2 < hi < 0.3)

    def test_split_groups_is_deterministic_and_stratified(self):
        groups = {f'i{i}': f'g{i // 2}' for i in range(20)}
        classes = {f'i{i}': D.RIPE if i < 12 else D.UNRIPE for i in range(20)}
        a = D.split_groups(groups, classes)
        self.assertEqual(a, D.split_groups(groups, classes))
        self.assertEqual(sorted(a[0] + a[1]), sorted(groups))
        self.assertTrue(any(classes[i] == D.UNRIPE for i in a[0]) and any(classes[i] == D.UNRIPE for i in a[1]))


if __name__ == '__main__':
    unittest.main()

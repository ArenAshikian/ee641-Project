"""Small tests for the external Maia benchmark. No models or external tools needed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
from chess_rl import maia_benchmark as mb
from chess_rl.maia_stats import compare_models, summarize_games


def game(model, idx, color, score, truncated=False):
    return {'model': model, 'maia': '1100', 'opening_id': idx, 'color': color,
            'score': score, 'truncated': truncated}


class TestMaiaStats(unittest.TestCase):
    def test_score_and_paired_ci(self):
        rows=[]
        for i in range(10):
            rows += [game('v0',i,'white',1.),game('v0',i,'black',0.)]
        report=summarize_games(rows,10,seed=11)
        self.assertEqual(report['wins'],10)
        self.assertEqual(report['losses'],10)
        self.assertEqual(report['complete_pairs'],10)
        self.assertEqual(report['score_completed'],.5)
        self.assertEqual(report['ci95_paired'],[.5,.5])

    def test_truncated_pair_not_falsely_drawn(self):
        rows=[game('v0',0,'white',1.),game('v0',0,'black',None,True),
              game('v0',1,'white',.5),game('v0',1,'black',.5)]
        report=summarize_games(rows,2)
        self.assertEqual(report['truncated'],1)
        self.assertEqual(report['complete_pairs'],1)
        self.assertAlmostEqual(report['score_completed'],2/3,places=5)
        self.assertIsNone(report['ci95_paired'])

    def test_model_comparison_on_matched_pair(self):
        a,b=[],[]
        for i in range(12):
            for color in ('white','black'):
                a.append(game('v100',i,color,1.))
                b.append(game('v0',i,color,0.))
        result=compare_models(a,b,seed=17)
        self.assertEqual(result['matched_opening_pairs'],12)
        self.assertEqual(result['score_delta'],1.)
        self.assertEqual(result['ci95_paired_difference'],[1.,1.])

    def test_partial_record_not_used_in_comparison(self):
        a=[game('v100',0,'white',1.),game('v100',0,'black',1.)]
        b=[game('v0',0,'white',0.)]
        self.assertIsNone(compare_models(a,b)['score_delta'])

    def test_append_resume_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'games.jsonl'
            record={'model':'v0','maia':'1100','opening_id':0,
                    'color':'white','score':.5,'truncated':False}
            mb.append_record(path,record)
            loaded=mb.read_records(path)
            self.assertEqual(len(loaded),1)
            self.assertEqual(loaded[('v0','1100',0,'white')]['score'],.5)
            mb.append_record(path,record)
            with self.assertRaises(RuntimeError):
                mb.read_records(path)

    def test_manifest_hash_detects_change(self):
        doc={'schema':1,'seed':7,'openings':[['e2e4','e7e5']],
             'source':{'path':'test','size':1,'modified_ns':1},
             'source_kind':'external_human_games','exclusions_sha256':'x'}
        before=mb.manifest_digest(doc)
        doc['seed']=8
        self.assertNotEqual(before,mb.manifest_digest(doc))

    def test_uci_one_node(self):
        class FakeEngine:
            def play(self, board, limit, info):
                self.limit=limit
                self.info=info
                return type('Result', (), {'move':chess.Move.from_uci('e2e4'),
                                           'info':{'nodes':1}})()
        fake=FakeEngine()
        move,nodes=mb.move_engine(fake,chess.Board())
        self.assertEqual(move.uci(),'e2e4')
        self.assertEqual(nodes,1)
        self.assertEqual(fake.limit.nodes,1)

    def test_uci_node_count_is_diagnostic(self):
        # policyhead guarantees one evaluation independently of Lc0's UCI
        # informational node counter, which can exceed the requested limit.
        class FakeEngine:
            def play(self, board, limit, info):
                self.limit=limit
                return type('Result', (), {'move':chess.Move.from_uci('e2e4'),
                                           'info':{'nodes':2}})()
        fake=FakeEngine()
        move,nodes=mb.move_engine(fake,chess.Board())
        self.assertEqual(move.uci(),'e2e4')
        self.assertEqual(nodes,2)
        self.assertEqual(fake.limit.nodes,1)

    def test_find_lc0_uses_policyhead_executable(self):
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/'lc0.exe'
            original.write_bytes(b'executable test content')
            with patch.object(mb, 'LC0', original), patch.dict('os.environ',{'LC0_PATH':''}):
                path=mb.find_lc0()
                self.assertEqual(path.name, 'lc0-policyhead.exe')
                self.assertEqual(path.read_bytes(),original.read_bytes())



if __name__ == '__main__':
    unittest.main()

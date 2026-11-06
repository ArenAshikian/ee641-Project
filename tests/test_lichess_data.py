import io
import unittest

import chess.pgn

from chess_rl.lichess_data import (positions_from_game, read_headers,
                                  read_raw_games, selection_reason, validation_game)


class TestLichessData(unittest.TestCase):
    def setUp(self):
        self.headers = {
            'Event': 'Rated Blitz game', 'Site': 'https://lichess.org/test123',
            'WhiteElo': '2000', 'BlackElo': '2100', 'TimeControl': '180+2',
            'Termination': 'Normal', 'Result': '1-0',
        }

    def test_game_filters(self):
        self.assertEqual(selection_reason(self.headers), 'accepted')
        self.assertEqual(selection_reason({**self.headers, 'WhiteTitle': 'BOT'}), 'bot_game')
        self.assertEqual(selection_reason({**self.headers, 'WhiteElo': '1700'}), 'rating_out_of_range')
        self.assertEqual(selection_reason({**self.headers, 'Event': 'Rated Bullet game'}), 'not_rated_blitz_or_rapid')
        self.assertEqual(selection_reason({**self.headers, 'Termination': 'Time forfeit'}), 'non_board_termination')

    def test_game_stream(self):
        raw1 = '[Event "Rated Blitz game"]\n[Site "game1"]\n\n1. e4 e5 *\n\n'
        raw2 = '[Event "Rated Rapid game"]\n[Site "game2"]\n\n1. d4 d5 *\n\n'
        games = list(read_raw_games(io.StringIO(raw1 + raw2)))
        self.assertEqual(len(games), 2)
        self.assertEqual(read_headers(games[1])['Site'], 'game2')

    def test_split_is_repeatable(self):
        self.assertEqual(validation_game(self.headers), validation_game(self.headers))

    def test_sample_move_and_result(self):
        pgn = '''[Event "Rated Blitz game"]
[Site "https://lichess.org/test123"]
[Date "2026.09.01"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2100"]
[TimeControl "180+2"]
[Termination "Normal"]

1. e4 { [%clk 0:02:58] [%eval 0.25] } 1... e5 { [%clk 0:02:58] } 2. Nf3 { [%clk 0:02:56] } 2... Nc6 { [%clk 0:02:56] } 1-0
'''
        game = chess.pgn.read_game(io.StringIO(pgn))
        samples = list(positions_from_game(game, max_positions_per_game=10, seed=1))
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0][0].shape, (61, 8, 8))
        self.assertEqual([s[2] for s in samples], [1.0, -1.0, 1.0, -1.0])

    def test_low_clock_excluded(self):
        pgn = '''[Event "Rated Blitz game"]
[Site "g"]
[Result "1-0"]

1. e4 { [%clk 0:00:20] } 1... e5 { [%clk 0:02:00] } 1-0
'''
        game = chess.pgn.read_game(io.StringIO(pgn))
        samples = list(positions_from_game(game, max_positions_per_game=10))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][2], -1.0)


if __name__ == '__main__':
    unittest.main()

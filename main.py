import io

import chess.pgn
import numpy
import torch
import zstandard
from torch.utils.data import DataLoader, TensorDataset


def read_game(stream):
    reading_comments = True
    white_elo = 0
    black_elo = 0
    result = -1  # 0:white 1:black 2:draw
    time_control = 0  # main time in seconds
    normal_termination = False

    count = 0
    while reading_comments and count < 50:
        line = stream.readline().strip()
        if line == "":
            reading_comments = False
        else:
            line = line[1:-1]
            line = line.split(" ", 1)
            key = line[0]
            if key == "WhiteElo":
                if line[1][1:-1].isnumeric():
                    white_elo = int(line[1][1:-1])
            elif key == "BlackElo":
                if line[1][1:-1].isnumeric():
                    black_elo = int(line[1][1:-1])
            elif key == "TimeControl":
                tc = line[1][1:-1]
                tc = tc.split("+")
                if tc[0].isnumeric():
                    time_control = int(tc[0])
            elif key == "Termination":
                normal_termination = line[1] == "\"Normal\""
            elif key == "Result":
                if line[1] == "\"1-0\"":
                    result = 0
                elif line[1] == "\"0-1\"":
                    result = 1
                elif line[1] == "\"1/2-1/2\"":
                    result = 2
        count += 1

    if white_elo > 2000 and black_elo > 2000 and result != -1 and time_control >= 180 and normal_termination:
        line = stream.readline().strip()
        return line
    else:
        stream.readline()
        stream.readline()
        return ""


def read_games_to_txt(input_path, output_path):
    with open(input_path, "rb") as input_file:
        print(input_file)
        with open(output_path, "w") as output_file:
            print(output_file)

            dctx = zstandard.ZstdDecompressor()
            reader = dctx.stream_reader(input_file)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            valid_games = 0
            for i in range(10000000):
                pgn = read_game(text_stream)
                if pgn != "":
                    valid_games += 1
                    s = "id_" + str(valid_games) + " " + pgn + "\n"
                    output_file.write(s)

            output_file.close()
        input_file.close()


def read_pgn_to_fens(input_path, output_path):
    with open(input_path, "r") as input_file:
        print(input_file)
        with open(output_path, "w") as output_file:
            print(output_file)

            fen_count = 0
            for i in range(535120):
                if i % 1000 == 0:
                    print(i)
                    print("fen_count", fen_count)
                line = input_file.readline()
                line = line.split(" ", 1)
                # id = line[0]
                pgn = line[1]
                pgn = io.StringIO(pgn)
                game = chess.pgn.read_game(pgn)
                result = game.headers["Result"]
                if result != "1/2-1/2":
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        output_file.write(board.fen() + "___" + result + "\n")
                        fen_count += 1
            print("fen_count_final", fen_count)


def read_fen_to_datasets(input_path, output_path, dataset_size, dataset_amount):
    with open(input_path, "r") as input_file:
        print(input_file)
        for dataset_id in range(dataset_amount):
            print(f"dataset_id is {dataset_id + 1}")

            dataset_input_list = []
            dataset_target_list = []

            for data_id in range(dataset_size):
                if data_id % 1000 == 0:
                    print(data_id)

                line = input_file.readline().strip()
                line = line.split("___", 1)
                fen, result = line[0], line[1]

                result_val = 0
                if result == "1-0":
                    result_val = 1
                elif result == "0-1":
                    result_val = -1

                target_tensor = torch.tensor(result_val, dtype=torch.float32)

                data_list = []

                board = chess.Board(fen)
                # COLORS = [WHITE, BLACK] = [True, False]
                # PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
                for pc_type in chess.PIECE_TYPES:
                    for color in chess.COLORS:
                        # print(pc_type)
                        # print(color)
                        pcs = board.pieces(piece_type=pc_type, color=color)
                        pcs = numpy.array(pcs.tolist()).astype(numpy.float32)
                        pcs = numpy.reshape(pcs, [8, 8])
                        # print(pcs)
                        data_list.append(pcs)

                turn = board.turn
                turn_board = numpy.full([8, 8], turn).astype(numpy.float32)
                data_list.append(turn_board)

                castling_rights = board.castling_rights
                castling_board = numpy.full([8, 8], False)
                castling_board[0][0] = bool(castling_rights & chess.BB_A8)
                castling_board[0][7] = bool(castling_rights & chess.BB_H8)
                castling_board[7][0] = bool(castling_rights & chess.BB_A1)
                castling_board[7][7] = bool(castling_rights & chess.BB_H1)
                castling_board = castling_board.astype(numpy.float32)
                data_list.append(castling_board)

                ep_square = board.ep_square
                legal_en_passant = board.has_legal_en_passant()
                if legal_en_passant:
                    ep_ss = chess.SquareSet(ep_square)
                    ep_board = numpy.array(ep_ss.tolist()).astype(numpy.float32)
                    ep_board = numpy.reshape(pcs, [8, 8])
                    data_list.append(ep_board)
                else:
                    ep_board = numpy.full([8, 8], 0, dtype=numpy.float32)
                    data_list.append(ep_board)

                data_tensor = torch.tensor(numpy.array(data_list))

                dataset_input_list.append(data_tensor)
                dataset_target_list.append(target_tensor)

            x_train = torch.tensor(numpy.array(dataset_input_list))
            y_train = torch.tensor(numpy.array(dataset_target_list))

            ds = TensorDataset(x_train, y_train)
            torch.save(ds, f"{output_path}/dataset_id_{dataset_id + 1}")


def main():
    # input_file_path = "Data/lichess_db_standard_rated_2024-01.pgn.zst"
    # output_file_path = "Data/filtered_chess_games_2024-01(535120).txt"
    # read_games_to_txt(input_file_path, output_file_path)

    # input_file_path = "Data/filtered_chess_games_2024-01(535120).txt"
    # output_file_path = "Data/no_draw_filtered_fen_2024-01(34418656).txt"
    # read_pgn_to_fens(input_file_path, output_file_path)

    input_file_path = "Data/no_draw_filtered_fen_2024-01(34418656).txt"
    output_folder_path = "Data/datasets_2024-01_no_draw"
    read_fen_to_datasets(input_file_path, output_folder_path, 2000000, 1)


if __name__ == "__main__":
    main()

from chess_positions import IllegalPosition
from chess_positions import Check
from glob import glob
from skimage.util.shape import view_as_blocks

import cv2 as cv
import numpy as np
import plotly.express as px
import random
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore')


class Pipeline(object):
    """
    This class is a pipeline mechanism to feed the
    query chess image into the model for FEN prediction.
    """

    def __init__(self, chess_image):
        self.piece_symbols = "prbnkqPRBNKQ"
        self.rows, self.cols = (8, 8)
        self.square = None
        self.h, self.w, self.c = None, None, None
        self.chess_image = chess_image
        self.chess_model = tf.keras.models.load_model(
            filepath='chess_model.h5')

    def preprocess(self, resize_scale=(200, 200)):
        """
        This method preprocesses the chess image.
        """
        img = cv.imread(filename=self.chess_image)
        img = cv.resize(src=img, dsize=resize_scale)

        self.h, self.w, self.c = img.shape
        self.square = self.h // self.rows

        img_blocks = view_as_blocks(
            arr_in=img, block_shape=(self.square, self.square, self.c))
        img_blocks = img_blocks.reshape(
            self.rows * self.cols, self.square, self.square, self.c)

        return img_blocks

    def fen_from_onehot(self, onehot):
        """
        This method converts onehot to FEN.
        The original author of this method is 'PAVEL KORYAKIN'.
        PAVEL KORYAKIN is also the maintainer of 'Chess Positions' dataset.
        """
        output = str()
        for j in range(self.rows):
            for i in range(self.cols):
                if onehot[j][i] == 12:
                    output += ' '
                else:
                    output += self.piece_symbols[int(onehot[j][i])]
            if j != 7:
                output += '/'

        for i in range(8, 0, -1):
            output = output.replace(' ' * i, str(i))

        return output

    def predict(self):
        """
        This method predicts the FEN of the query chess image.
        """
        chess_image_blocks = self.preprocess()

        onehot = self.chess_model.predict(x=chess_image_blocks)
        onehot = onehot.argmax(axis=1).reshape(-1, 8, 8)[0]

        fen_label = self.fen_from_onehot(onehot=onehot)

        interpretation = self.illegal_interpreter(fen_label=fen_label)
        if len(interpretation) > 0:
            interpretation = "This is an illegal chess position. Reason is " + interpretation
        else:
            interpretation = self.check_interpreter(fen_label=fen_label)
        
        return fen_label, interpretation

    def illegal_interpreter(self, fen_label):
        """
        This method interprets the predicted FEN.
        """
        reason = str()

        chess_illegal = IllegalPosition(fen_label=fen_label)

        if chess_illegal.are_kings_less():
            reason += "either white king, black king, or both are missing."
        elif chess_illegal.are_kings_more():
            reason += "either white king, black king, or both are more than 1."
        elif chess_illegal.are_queens_more():
            reason += "either white queen, black queen, or both are more than 9."
        elif chess_illegal.are_bishops_more():
            reason += "either white bishop, black bishop, or both are more than 10."
        elif chess_illegal.are_knights_more():
            reason += "either white knight, black knight, or both are more than 10."
        elif chess_illegal.are_rooks_more():
            reason += "either white rook, black rook, or both are more than 10."
        elif chess_illegal.are_pawns_more():
            reason += "either white pawn, black pawn, or both are more than 8."
        elif chess_illegal.rule_2():
            reason += "either white pawn, black pawn, or both are in the first row and/or the last row."
        elif chess_illegal.rule_3():
            reason += "the king checks the other the king."
        elif chess_illegal.rule_4():
            reason += "white king and black king are under attack simultaneously."
        else:
            reason += ""

        return reason

    def check_interpreter(self, fen_label):
        """
        This method interprets the predicted FEN.
        """
        reason = str()

        chess_check = Check(fen_label=fen_label)

        r_checks_K = chess_check.rook_checks_king(
            attacker='r', defendant='K')
        n_checks_K = chess_check.knight_checks_king(
            attacker='n', defendant='K')
        b_checks_K = chess_check.bishop_checks_king(
            attacker='b', defendant='K')
        q_checks_K = chess_check.queen_checks_king(
            attacker='q', defendant='K')
        p_checks_K = chess_check.pawn_checks_king(
            attacker='p', defendant='K')
        R_checks_k = chess_check.rook_checks_king(
            attacker='R', defendant='k')
        N_checks_k = chess_check.knight_checks_king(
            attacker='N', defendant='k')
        B_checks_k = chess_check.bishop_checks_king(
            attacker='B', defendant='k')
        Q_checks_k = chess_check.queen_checks_king(
            attacker='Q', defendant='k')
        P_checks_k = chess_check.pawn_checks_king(
            attacker='P', defendant='k')

        is_K_checked = r_checks_K or n_checks_K or b_checks_K or q_checks_K or p_checks_K
        is_k_checked = R_checks_k or N_checks_k or B_checks_k or Q_checks_k or P_checks_k

        if is_K_checked:
            reason += "The white king is under attack."
        elif is_k_checked:
            reason += "The black king is under attack."
        else:
            reason += "Both kings are safe."

        return reason


if __name__ == '__main__':
    chess_image = random.choice(glob(pathname=('./dataset/test/*jpeg')))
    print(chess_image)
    pipe = Pipeline(chess_image=chess_image)
    fen_label, interpretation = pipe.predict()
    print(fen_label)
    print(interpretation)

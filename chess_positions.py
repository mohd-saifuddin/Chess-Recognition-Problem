import re
import numpy as np


class AttackPositions(object):
    """
    This class is for finding an attack chess position in a board.
    """

    def __init__(self, fen_label):
        self.fen_label = re.sub(pattern=r'\d',
                                repl=lambda x: self.get_zeros(char=x.group()),
                                string=fen_label)
        self.fen_matrix = self.get_fen_matrix()

    def get_zeros(self, char):
        """
        This method returns repetitive 0s based on input character.
        """
        if char.isdigit():
            return '0' * int(char)

    def get_fen_matrix(self):
        """
        This method constructs a FEN matrix.
        """
        fen_matrix = np.array([list(row) for row in self.fen_label.split('/')])
        return fen_matrix

    def get_piece_positions(self, notation):
        """
        This method returns the 2D index of the piece from FEN matrix.
        """
        (i, j) = np.where(self.fen_matrix == notation)
        try:
            if i is not None and j is not None:
                return i, j
        except:
            return None

    def get_sub_matrix(self, ai, aj, di, dj):
        """
        This method chops the chessboard to a sub-matrix.
        """
        corners = np.array([(ai, aj), (di, aj), (ai, dj), (di, dj)])
        min_i, max_i = min(corners[:, 0]), max(corners[:, 0])
        min_j, max_j = min(corners[:, 1]), max(corners[:, 1])
        sub_matrix = self.fen_matrix[min_i:max_i+1, min_j:max_j+1]
        return sub_matrix, sub_matrix.shape

    def get_straight_checks(self, ai, aj, di, dj, a, d):
        """
        This method returns the checks along the straight path.
        """
        checks = list()
        for (i, j) in zip(ai, aj):
            if di == i:
                attack_path = self.fen_matrix[di]
            elif dj == j:
                attack_path = self.fen_matrix[:, dj]
            else:
                continue
            a_ind = np.where(attack_path == a)[0][0]
            d_ind = np.where(attack_path == d)[0][0]
            attack_path = attack_path[min(a_ind, d_ind): max(a_ind, d_ind)+1]
            checks.append(np.where(attack_path != '0')[0])
        checks = list(filter(lambda x: len(x) == 2, checks))
        return checks

    def get_diagonal_checks(self, ai, aj, di, dj, a):
        """
        This method returns the checks along the diagonal path.
        """
        checks = list()
        for (i, j) in zip(ai, aj):
            sub_mat, sub_shape = self.get_sub_matrix(ai=i, aj=j, di=di, dj=dj)
            if sub_shape[0] == sub_shape[1]:
                if a not in sub_mat.diagonal():
                    sub_mat = np.flipud(m=sub_mat)
                checks.append(np.where(sub_mat.diagonal() != '0')[0])
            else:
                continue
        checks = list(filter(lambda x: len(x) == 2, checks))
        return checks

    def get_l_shaped_checks(self, ai, aj, di, dj):
        """
        This method returns the checks along the L-shaped paths for knights.
        """
        checks = list()
        for (i, j) in zip(ai, aj):
            attack_positions = [(i-2, j-1), (i-2, j+1),
                                (i-1, j-2), (i-1, j+2),
                                (i+1, j-2), (i+1, j+2),
                                (i+2, j-1), (i+2, j+1)]
            if (di, dj) in attack_positions:
                checks.append((i, j))
        return checks

    def get_pawn_checks(self, ai, aj, di, dj):
        """
        This method returns the checks for pawns.
        """
        checks = list()
        for (i, j) in zip(ai, aj):
            _, sub_shape = self.get_sub_matrix(ai=i, aj=j, di=di, dj=dj)
            if sub_shape[0] == 2 and sub_shape[1] == 2:
                checks.append((i, j))
            else:
                continue
        return checks

    def king_attacks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the other king.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        di, dj = di[0], dj[0]
        ai, aj = self.get_piece_positions(notation=attacker)
        ai, aj = ai[0], aj[0]
        attack_positions = [(di, dj-1), (di, dj+1),
                            (di-1, dj), (di+1, dj),
                            (di-1, dj+1), (di-1, dj-1),
                            (di+1, dj-1), (di+1, dj+1)]
        if (ai, aj) in attack_positions:
            flag = True
        return flag

    def rook_attacks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the rook.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        di, dj = di[0], dj[0]
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_straight_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker, d=defendant)
        if checks:
            flag = True
        return flag

    def bishop_attacks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the bishop.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        di, dj = di[0], dj[0]
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_diagonal_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker)
        if checks:
            flag = True
        return flag

    def knight_attacks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the knight.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        di, dj = di[0], dj[0]
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_l_shaped_checks(ai=ai, aj=aj, di=di, dj=dj)
        if checks:
            flag = True
        return flag

    def queen_attacks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the queen.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        di, dj = di[0], dj[0]
        ai, aj = self.get_piece_positions(notation=attacker)
        straight_checks = self.get_straight_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker, d=defendant)
        diagonal_checks = self.get_diagonal_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker)
        if straight_checks or diagonal_checks:
            flag = True
        return flag

    def pawn_attacks_king(self, attacker, defendant):
        """
        This methos checks if the king is being attacked by the pawn.

        Note: It is hard to determine from an image, which side of 
              the chessboard is black or is white.
              Hence, this method assumes the pawn is attacking the king 
              if both are diagnolly aligned by 1 step.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        di, dj = di[0], dj[0]
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_pawn_checks(ai=ai, aj=aj, di=di, dj=dj)
        if checks:
            flag = True
        return flag

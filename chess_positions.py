import re
import numpy as np


class Check(object):
    """
    This class is for finding an attack chess position in a board.
    """

    def __init__(self, fen_label):
        self.fen_label = re.sub(pattern=r'\d',
                                repl=lambda x: self.get_ones(char=x.group()),
                                string=fen_label)
        self.fen_matrix = self.get_fen_matrix()

    def get_ones(self, char):
        """
        This method returns repetitive 0s based on input character.
        """
        if char.isdigit():
            return '1' * int(char)

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
            checks.append(np.where(attack_path != '1')[0])
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
                checks.append(np.where(sub_mat.diagonal() != '1')[0])
            else:
                continue
        checks = list(filter(lambda x: len(x) == 2, checks))
        return checks

    def get_knight_checks(self, ai, aj, di, dj):
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

    def king_checks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the other king.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) != 0 and len(dj) != 0:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        ai, aj = ai[0], aj[0]
        attack_positions = [(di, dj-1), (di, dj+1),
                            (di-1, dj), (di+1, dj),
                            (di-1, dj+1), (di-1, dj-1),
                            (di+1, dj-1), (di+1, dj+1)]
        if (ai, aj) in attack_positions:
            flag = True
        return flag

    def rook_checks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the rook.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) != 0 and len(dj) != 0:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_straight_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker, d=defendant)
        if checks:
            flag = True
        return flag

    def bishop_checks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the bishop.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) != 0 and len(dj) != 0:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_diagonal_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker)
        if checks:
            flag = True
        return flag

    def knight_checks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the knight.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) != 0 and len(dj) != 0:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_knight_checks(ai=ai, aj=aj, di=di, dj=dj)
        if checks:
            flag = True
        return flag

    def queen_checks_king(self, attacker, defendant):
        """
        This method checks if the king is being attacked by the queen.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) != 0 and len(dj) != 0:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        straight_checks = self.get_straight_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker, d=defendant)
        diagonal_checks = self.get_diagonal_checks(
            ai=ai, aj=aj, di=di, dj=dj, a=attacker)
        if straight_checks or diagonal_checks:
            flag = True
        return flag

    def pawn_checks_king(self, attacker, defendant):
        """
        This methos checks if the king is being attacked by the pawn.

        Note: It is hard to determine from an image, which side of 
              the chessboard is black or is white.
              Hence, this method assumes the pawn is attacking the king 
              if both are diagnolly aligned by 1 step.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) != 0 and len(dj) != 0:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_pawn_checks(ai=ai, aj=aj, di=di, dj=dj)
        if checks:
            flag = True
        return flag


class IllegalPositions(Check):
    """
    This class attempts to find if the chess pieces are illegally.
    """

    def __init__(self, fen_label):
        super().__init__(fen_label)

    def get_count_of_piece(self, notation):
        """
        This method returns the count of a piece in a chessboard.
        """
        piece_count = self.fen_label.count(notation)
        return piece_count

    def rule_1(self):
        """
        This method checks the count of the kings and the pieces in the board.
        1. The count of white king and black king should always be 1.
        2. The count of white queen and black queen should not cross 9.
        3. The count of white bishop and black bishop should not cross 10.
        4. The count of white knight and black knight should not cross 10.
        5. The count of white rook and black rook should not cross 10.
        6. The count of while pawn and black pawn should not cross 8.
        7. The chessboard should never be empty.
        """
        flag = False
        k_count = self.get_count_of_piece(notation='k')
        K_count = self.get_count_of_piece(notation='K')
        q_count = self.get_count_of_piece(notation='q')
        Q_count = self.get_count_of_piece(notation='Q')
        b_count = self.get_count_of_piece(notation='b')
        B_count = self.get_count_of_piece(notation='B')
        n_count = self.get_count_of_piece(notation='n')
        N_count = self.get_count_of_piece(notation='N')
        r_count = self.get_count_of_piece(notation='r')
        R_count = self.get_count_of_piece(notation='R')
        p_count = self.get_count_of_piece(notation='p')
        P_count = self.get_count_of_piece(notation='P')
        if k_count > 1 or k_count <= 0 or K_count > 1 or K_count <= 0:
            flag = True
        elif q_count > 9 or Q_count > 9:
            flag = True
        elif b_count > 10 or B_count > 10:
            flag = True
        elif n_count > 10 or N_count > 10:
            flag = True
        elif r_count > 10 or R_count > 10:
            flag = True
        elif p_count > 8 or P_count > 8:
            flag = True
        return flag

    def rule_2(self):
        """
        This method checks if the pawns are in the first and last row of the board.
        1. No pawn should be on the first row and on the last row.
           The pawn that reaches the last row always gets promoted.
           Hence no pawns on the last row.
        """
        flag = False
        fen_label_list = self.fen_label.split('/')
        f_row, l_row = fen_label_list[0], fen_label_list[-1]
        if 'p' in f_row or 'p' in l_row:
            flag = True
        elif 'P' in f_row or 'P' in l_row:
            flag = True
        return flag

    def rule_3(self):
        """
        This method checks if the king is attacking the other king.
        1. The king never checks the other king.
        2. The king can attack other enemy pieces except the enemy king.
        """
        flag = False
        if self.king_checks_king(attacker='k', defendant='K'):
            flag = True
        return flag

    def rule_4(self):
        """
        This method checks if the kings are under check simultaneously.
        1. The two kings are never under check at the same time.
           It is illegal.
        """
        flag = False
        r_checks_K = self.rook_checks_king(attacker='r', defendant='K')
        n_checks_K = self.knight_checks_king(attacker='n', defendant='K')
        b_checks_K = self.bishop_checks_king(attacker='b', defendant='K')
        q_checks_K = self.queen_checks_king(attacker='q', defendant='K')
        p_checks_K = self.pawn_checks_king(attacker='p', defendant='K')
        R_checks_k = self.rook_checks_king(attacker='R', defendant='k')
        N_checks_k = self.knight_checks_king(attacker='N', defendant='k')
        B_checks_k = self.bishop_checks_king(attacker='B', defendant='k')
        Q_checks_k = self.queen_checks_king(attacker='Q', defendant='k')
        P_checks_k = self.pawn_checks_king(attacker='P', defendant='k')
        is_K_checked = r_checks_K or n_checks_K or b_checks_K or q_checks_K or p_checks_K
        is_k_checked = R_checks_k or N_checks_k or B_checks_k or Q_checks_k or P_checks_k
        if is_K_checked and is_k_checked:
            flag = True
        return flag

    def is_illegal(self):
        """
        This method is a consolidation of all the above basic rules of chess.
        """
        flag = False
        r1 = self.rule_1()
        r2 = self.rule_2()
        r3 = self.rule_3()
        r4 = self.rule_4()
        if r1:
            # print("Rule 1")
            flag = True
        elif r2:
            # print("Rule 2")
            flag = True
        elif r3:
            # print("Rule 3")
            flag = True
        elif r4:
            # print("Rule 4")
            flag = True
        return flag

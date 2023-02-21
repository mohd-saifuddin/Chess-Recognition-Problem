import re
import numpy as np


class Board(object):
    """
    This class is defines the chessboard.
    """

    def __init__(self, fen_label):
        self.fen_label = re.sub(pattern=r'\d',
                                repl=lambda x: self.get_ones(char=x.group()),
                                string=fen_label)
        self.fen_matrix = self.get_fen_matrix()

    def get_ones(self, char):
        """
        This method returns repetitive 1s based on input digit character.
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


class Check(Board):
    """
    This class finds if there are any checks in the chessboard.
    """

    def __init__(self, fen_label):
        super().__init__(fen_label=fen_label)

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
        This is unlikely, but I am just adding a validation rule.
        """
        flag = False
        di, dj = self.get_piece_positions(notation=defendant)
        if len(di) == 1 and len(dj) == 1:
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
        if len(di) == 1 and len(dj) == 1:
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
        if len(di) == 1 and len(dj) == 1:
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
        if len(di) == 1 and len(dj) == 1:
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
        if len(di) == 1 and len(dj) == 1:
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
        if len(di) == 1 and len(dj) == 1:
            di, dj = di[0], dj[0]
        else:
            return flag
        ai, aj = self.get_piece_positions(notation=attacker)
        checks = self.get_pawn_checks(ai=ai, aj=aj, di=di, dj=dj)
        if checks:
            flag = True
        return flag


class IllegalPosition(Check):
    """
    This class finds if the pieces are illegally positioned in the chessboard.
    """

    def __init__(self, fen_label):
        super().__init__(fen_label=fen_label)

    def are_kings_less(self):
        """
        Rule on kings.
        """
        k_c = self.fen_label.count('k')
        K_c = self.fen_label.count('K')
        return (k_c < 1 and K_c < 1) or (k_c < 1) or (K_c < 1)

    def are_kings_more(self):
        """
        Rule on kings.
        """
        k_c = self.fen_label.count('k')
        K_c = self.fen_label.count('K')
        return (k_c > 1 and K_c > 1) or (k_c > 1) or (K_c > 1)

    def are_queens_more(self):
        """
        Rule on queens.
        """
        q_c = self.fen_label.count('q')
        Q_c = self.fen_label.count('Q')
        return (q_c > 9 and Q_c > 9) or (q_c > 9) or (Q_c > 9)

    def are_bishops_more(self):
        """
        Rule on bishops.
        """
        b_c = self.fen_label.count('b')
        B_c = self.fen_label.count('B')
        return (b_c > 10 and B_c > 10) or (b_c > 10) or (B_c > 10)

    def are_knights_more(self):
        """
        Rule on knights.
        """
        n_c = self.fen_label.count('n')
        N_c = self.fen_label.count('N')
        return (n_c > 10 and N_c > 10) or (n_c > 10) or (N_c > 10)

    def are_rooks_more(self):
        """
        Rule on rooks.
        """
        r_c = self.fen_label.count('r')
        R_c = self.fen_label.count('R')
        return (r_c > 10 and R_c > 10) or (r_c > 10) or (R_c > 10)

    def are_pawns_more(self):
        """
        Rule on pawns.
        """
        p_c = self.fen_label.count('p')
        P_c = self.fen_label.count('P')
        return (p_c > 8 and P_c > 8) or (p_c > 8) or (P_c > 8)

    def rule_1(self):
        """
        This method checks the count of the kings and the pieces in the board.
        1. The count of white king and black king should always be 1.
        2. The count of white queen and/or black queen should not cross 9.
        3. The count of white bishop and/or black bishop should not cross 10.
        4. The count of white knight and/or black knight should not cross 10.
        5. The count of white rook and/or black rook should not cross 10.
        6. The count of while pawn and/or black pawn should not cross 8.
        7. The chessboard should never be empty.
        """
        flag = False
        if self.are_kings_less():
            flag = True
        elif self.are_kings_more():
            flag = True
        elif self.are_queens_more():
            flag = True
        elif self.are_bishops_more():
            flag = True
        elif self.are_knights_more():
            flag = True
        elif self.are_rooks_more():
            flag = True
        elif self.are_pawns_more():
            flag = True
        return flag

    def rule_2(self):
        """
        This method checks if the pawns are in the first and last row of the board.
        1. No pawn should be on the first row and/or on the last row.
           The pawn that reaches the last row always gets promoted.
           Hence no pawns on the last row.
        """
        flag = False
        fen_label_list = self.fen_label.split('/')
        f_row, l_row = fen_label_list[0], fen_label_list[-1]
        p_f_row = 'p' in f_row
        p_l_row = 'p' in l_row
        P_f_row = 'P' in f_row
        P_l_row = 'P' in l_row
        if (p_f_row and p_l_row) or p_f_row or p_l_row:
            flag = True
        elif (P_f_row and P_l_row) or P_f_row or P_l_row:
            flag = True
        return flag

    def rule_3(self):
        """
        This method checks if the king is attacking the other king.
        1. The king never checks the other king.
        2. The king can attack other enemy pieces except the enemy king.
        """
        return self.king_checks_king(attacker='k', defendant='K')

    def rule_4(self):
        """
        This method checks if the kings are under check simultaneously.
        1. The two kings are never under check at the same time.
        """
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
        return is_K_checked and is_k_checked

    def is_illegal(self):
        """
        This method is a consolidation of all the above basic rules of chess.
        """
        return self.rule_1() or self.rule_2() or self.rule_3() or self.rule_4()

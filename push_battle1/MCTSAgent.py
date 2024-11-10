import random
import math
import time
from PushBattle import Game, PLAYER1, PLAYER2, EMPTY, BOARD_SIZE, NUM_PIECES, _torus
import copy

class MCTSAgent:
    def __init__(self, player=PLAYER1):
        self.player = player
        self.exploration_weight = 1.0
        self.simulation_time = 1.0
        
    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state"""
        game_copy = self.clone_game(game)  # Work with a copy
        moves = []
        current_pieces = game_copy.p1_pieces if game_copy.current_player == PLAYER1 else game_copy.p2_pieces
        
        if current_pieces < NUM_PIECES:
            # placement moves
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game_copy.board[r][c] == EMPTY:
                        if game_copy.is_valid_placement(r, c):
                            moves.append((r, c))
        else:
            # movement moves
            for r0 in range(BOARD_SIZE):
                for c0 in range(BOARD_SIZE):
                    if game_copy.board[r0][c0] == game_copy.current_player:
                        for r1 in range(BOARD_SIZE):
                            for c1 in range(BOARD_SIZE):
                                if game_copy.board[r1][c1] == EMPTY:
                                    if game_copy.is_valid_move(r0, c0, r1, c1):
                                        moves.append((r0, c0, r1, c1))
        return moves

    def clone_game(self, game):
        """Create a deep copy of the game state"""
        new_game = copy.deepcopy(game) 
        return new_game

    def simulate_move(self, game, move):
        """Simulate a move on a copy of the game"""
        game_copy = self.clone_game(game)
        try:
            if len(move) == 2:
                if game_copy.is_valid_placement(*move):
                    game_copy.place_checker(*move)
                    return game_copy
            else:
                if game_copy.is_valid_move(*move):
                    game_copy.move_checker(*move)
                    return game_copy
        except:
            pass
        return None


    def get_best_move(self, game):
        """Returns best move using MCTS with UCB1"""
        possible_moves = self.get_possible_moves(game)
        if not possible_moves:
            return None
            
        # Get valid moves
        valid_moves = []
        for move in possible_moves:
            game_copy = self.clone_game(game)
            try:
                if len(move) == 2:
                    if game_copy.is_valid_placement(*move):
                        valid_moves.append(move)
                else:
                    if game_copy.is_valid_move(*move):
                        valid_moves.append(move)
            except:
                continue
        
        if not valid_moves:
            return None

        # Run MCTS
        move_stats = {}  # move -> (total_score, visits)
        exploration_constant = 1.414  # sqrt(2)
        end_time = time.time() + 0.95
        
        while time.time() < end_time:
            # Select move using UCB1
            total_visits = sum(visits for _, visits in move_stats.values()) + 1
            selected_move = None
            best_ucb = float('-inf')
            
            for move in valid_moves:
                total_score, visits = move_stats.get(move, (0, 0))
                
                if visits == 0:
                    selected_move = move
                    break
                    
                # UCB1 calculation
                exploitation = total_score / visits if visits > 0 else 0
                exploration = exploration_constant * math.sqrt(math.log(total_visits) / visits)
                ucb = exploitation + exploration
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    selected_move = move
            
            # If all moves tried, pick one with highest UCB
            if selected_move is None:
                selected_move = random.choice(valid_moves)
                
            # Simulate game
            score = self.simulate_game(game, selected_move)
            
            # Update statistics
            current_score, visits = move_stats.get(selected_move, (0, 0))
            move_stats[selected_move] = (current_score + score, visits + 1)
        
        # Select best move based on average score
        best_move = None
        best_average = float('-inf')
        
        for move in valid_moves:
            total_score, visits = move_stats.get(move, (0, 0))
            if visits > 0:
                average = total_score / visits
                if average > best_average:
                    best_average = average
                    best_move = move
        
        return best_move if best_move is not None else random.choice(valid_moves)



    def count_aligned_pieces(self, game, player):
        """Count number of 2-in-a-row configurations"""
        score = 0
        # Horizontal
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE-1):
                if game.board[r][c] == game.board[r][c+1] == player:
                    score += 10
        
        # Vertical
        for r in range(BOARD_SIZE-1):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == game.board[r+1][c] == player:
                    score += 10
                    
        return score

    def evaluate_center_control(self, game, player):
        """Evaluate control of center squares"""
        score = 0
        center_squares = [
            (3,3), (3,4), (4,3), (4,4)  # Center 2x2
        ]
        
        for r, c in center_squares:
            if game.board[r][c] == player:
                score += 5
                
        return score

    def evaluate_protected_pieces(self, game, player):
        """Evaluate how well pieces are protected from pushes"""
        score = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == player:
                    if self.is_protected(game, r, c, player):
                        score += 3
                        
        return score

    def is_protected(self, game, r, c, player):
        """Check if a piece is protected from pushes"""
        # Check all four directions
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            # Position that could push
            push_r = (r + dr) % BOARD_SIZE
            push_c = (c + dc) % BOARD_SIZE
            
            # Position behind (for protection)
            behind_r = (r - dr) % BOARD_SIZE
            behind_c = (c - dc) % BOARD_SIZE
            
            # If there's a threat and no protection
            if (game.board[push_r][push_c] != EMPTY and 
                game.board[push_r][push_c] != player and
                game.board[behind_r][behind_c] != player):
                return False
                
        return True

    def simulate_game(self, game, first_move):
        """Simulate a game with heuristic evaluation"""
        sim_game = self.clone_game(game)
        try:
            # Make first move
            if len(first_move) == 2:
                sim_game.place_checker(*first_move)
            else:
                sim_game.move_checker(*first_move)
                
            sim_game.current_player *= -1
            moves_count = 1
            max_moves = 50  # Reduced from 100 for faster simulations
            
            while moves_count < max_moves:
                winner = sim_game.check_winner()
                if winner != EMPTY:
                    return 1.0 if winner == self.player else -1.0
                    
                possible_moves = self.get_possible_moves(sim_game)
                valid_moves = []
                move_scores = []  # Store move evaluations
                
                # Evaluate each possible move
                for move in possible_moves:
                    game_copy = self.clone_game(sim_game)
                    try:
                        if len(move) == 2:
                            if game_copy.is_valid_placement(*move):
                                game_copy.place_checker(*move)
                                valid_moves.append(move)
                                score = self.evaluate_position(game_copy, sim_game.current_player)
                                move_scores.append(score)
                        else:
                            if game_copy.is_valid_move(*move):
                                game_copy.move_checker(*move)
                                valid_moves.append(move)
                                score = self.evaluate_position(game_copy, sim_game.current_player)
                                move_scores.append(score)
                    except:
                        continue
                
                if not valid_moves:
                    return 0.0
                    
                # Choose move based on scores with some randomness
                if random.random() < 0.8:  # 80% choose best move
                    best_idx = move_scores.index(max(move_scores))
                    move = valid_moves[best_idx]
                else:  # 20% random for exploration
                    move = random.choice(valid_moves)
                    
                if len(move) == 2:
                    sim_game.place_checker(*move)
                else:
                    sim_game.move_checker(*move)
                    
                sim_game.current_player *= -1
                moves_count += 1
                
            # If game hasn't ended, evaluate final position
            return self.evaluate_position(sim_game, self.player) / 1000.0
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return 0.0


    def choose_move(self, valid_moves, move_stats):
        """Choose a move to explore using UCB1"""
        total_plays = sum(plays for _, plays in move_stats.values())
        
        # If some moves haven't been tried, try them first
        untried = [move for move in valid_moves if move not in move_stats]
        if untried:
            return random.choice(untried)
        
        # Otherwise use UCB1 formula
        best_score = -float('inf')
        best_moves = []
        
        for move in valid_moves:
            wins, plays = move_stats.get(move, (0, 0))
            if plays == 0:
                continue
                
            # UCB1 calculation
            score = wins/plays + math.sqrt(2 * math.log(total_plays) / plays)
            
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)
        
        return random.choice(best_moves)



    def evaluate_position(self, game, player):
        """Enhanced position evaluation"""
        score = 0
        
        # Check for immediate win
        winner = game.check_winner()
        if winner == player:
            return 1000
        elif winner == -player:
            return -1000
        
        # Check for winning threats
        threat_score = self.evaluate_threats(game, player)
        score += threat_score * 50  # High priority for threats
        
        # Check for winning patterns
        pattern_score = self.evaluate_patterns(game, player)
        score += pattern_score * 30
        
        # Previous evaluations
        score += self.count_aligned_pieces(game, player)
        score += self.evaluate_center_control(game, player)
        score += self.evaluate_protected_pieces(game, player)
        
        return score

    def evaluate_threats(self, game, player):
        """Evaluate immediate threats and potential wins"""
        score = 0
        opponent = -player
        
        # Check for two-in-a-row with empty third position
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == player:
                    for dr, dc in directions:
                        threat = self.check_threat(game, r, c, dr, dc, player)
                        score += threat
                        
                        # Check if opponent has similar threat
                        opp_threat = self.check_threat(game, r, c, dr, dc, opponent)
                        score -= opp_threat * 1.2  # Weigh opponent threats slightly higher
        
        return score

    def check_threat(self, game, r, c, dr, dc, player):
        """Check for specific threat patterns"""
        try:
            # Get the three positions in line
            pos1 = game.board[r][c]
            pos2 = game.board[(r + dr) % BOARD_SIZE][(c + dc) % BOARD_SIZE]
            pos3 = game.board[(r + 2*dr) % BOARD_SIZE][(c + 2*dc) % BOARD_SIZE]
            
            # Two pieces with empty third position
            if pos1 == pos2 == player and pos3 == EMPTY:
                # Check if the empty position can be reached
                if self.is_position_reachable(game, (r + 2*dr) % BOARD_SIZE, 
                                            (c + 2*dc) % BOARD_SIZE, player):
                    return 15
                    
            # Piece-empty-piece pattern
            if pos1 == pos3 == player and pos2 == EMPTY:
                if self.is_position_reachable(game, (r + dr) % BOARD_SIZE, 
                                            (c + dc) % BOARD_SIZE, player):
                    return 10
                    
        except IndexError:
            pass
        return 0

    def is_position_reachable(self, game, target_r, target_c, player):
        """Check if a position can be reached by the player"""
        # If in placement phase
        current_pieces = game.p1_pieces if player == PLAYER1 else game.p2_pieces
        if current_pieces < NUM_PIECES:
            return True
            
        # In movement phase, check if any piece can move there
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == player:
                    game_copy = self.clone_game(game)
                    try:
                        if game_copy.is_valid_move(r, c, target_r, target_c):
                            return True
                    except:
                        continue
        return False

    def evaluate_patterns(self, game, player):
        """Evaluate board patterns"""
        score = 0
        
        # Triangle pattern (strong defensive formation)
        score += self.find_triangle_patterns(game, player) * 20
        
        # Wall pattern (line of protected pieces)
        score += self.find_wall_patterns(game, player) * 15
        
        # Fork pattern (multiple threats)
        score += self.find_fork_patterns(game, player) * 25
        
        return score

    def find_triangle_patterns(self, game, player):
        """Find triangle formations of pieces"""
        count = 0
        for r in range(BOARD_SIZE-1):
            for c in range(BOARD_SIZE-1):
                if (game.board[r][c] == player and
                    game.board[r+1][c] == player and
                    game.board[r][c+1] == player):
                    count += 1
                    # Check if protected
                    if all(self.is_protected(game, r+dr, c+dc, player) 
                        for dr, dc in [(0,0), (1,0), (0,1)]):
                        count += 1
        return count

    def find_wall_patterns(self, game, player):
        """Find wall formations (protected lines)"""
        count = 0
        # Horizontal walls
        for r in range(BOARD_SIZE):
            wall_length = 0
            for c in range(BOARD_SIZE):
                if (game.board[r][c] == player and 
                    self.is_protected(game, r, c, player)):
                    wall_length += 1
                else:
                    if wall_length >= 2:
                        count += wall_length - 1
                    wall_length = 0
        
        # Vertical walls
        for c in range(BOARD_SIZE):
            wall_length = 0
            for r in range(BOARD_SIZE):
                if (game.board[r][c] == player and 
                    self.is_protected(game, r, c, player)):
                    wall_length += 1
                else:
                    if wall_length >= 2:
                        count += wall_length - 1
                    wall_length = 0
        return count

    def find_fork_patterns(self, game, player):
        """Find fork patterns (multiple threats)"""
        count = 0
        threat_positions = set()
        
        # Find all positions that would complete a two-in-a-row
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if game.board[r][c] == EMPTY:
                    threats = 0
                    # Check all directions
                    for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                        if self.would_create_threat(game, r, c, dr, dc, player):
                            threats += 1
                    if threats >= 2:  # Position creates multiple threats
                        count += 1
                        threat_positions.add((r, c))
        
        # Bonus for reachable threat positions
        for r, c in threat_positions:
            if self.is_position_reachable(game, r, c, player):
                count += 1
                
        return count

    def would_create_threat(self, game, r, c, dr, dc, player):
        """Check if placing at (r,c) would create a threat"""
        try:
            # Check both directions
            pos1 = game.board[(r + dr) % BOARD_SIZE][(c + dc) % BOARD_SIZE]
            pos2 = game.board[(r - dr) % BOARD_SIZE][(c - dc) % BOARD_SIZE]
            
            # Would create two in a row
            if pos1 == player or pos2 == player:
                return True
                
        except IndexError:
            pass
        return False

import random
import math
import time
import copy

class FastMCTSAgent:
    def __init__(self, player=1):
        self.player = player
        self.exploration_weight = 1.0
        self.time_limit = 0.95  # Slightly less than 1 second to account for overhead
        
    def get_possible_moves(self, game):
        """Returns list of all possible moves in current state"""
        moves = []
        current_pieces = game.p1_pieces if game.current_player == 1 else game.p2_pieces
        
        if current_pieces < 8:  # Using constant instead of NUM_PIECES
            # Placement moves - only check empty spaces
            for r in range(8):  # Using constant instead of BOARD_SIZE
                for c in range(8):
                    if game.board[r][c] == 0:  # Using 0 instead of EMPTY
                        moves.append((r, c))
        else:
            # Movement moves - only check player's pieces to empty spaces
            for r0 in range(8):
                for c0 in range(8):
                    if game.board[r0][c0] == game.current_player:
                        # Only check adjacent spaces for movement
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            r1 = (r0 + dr) % 8
                            c1 = (c0 + dc) % 8
                            if game.board[r1][c1] == 0:
                                moves.append((r0, c0, r1, c1))
        return moves

    def clone_game(self, game):
        """Create a lightweight copy of the game state"""
        return copy.deepcopy(game)

    def get_best_move(self, game):
        """Returns best move using optimized MCTS"""
        possible_moves = self.get_possible_moves(game)
        if not possible_moves:
            return None

        # Initialize move statistics
        move_stats = {}  # move -> (total_score, visits)
        end_time = time.time() + self.time_limit
        exploration_constant = math.sqrt(2)
        
        # Quick evaluation of all moves
        move_priorities = {}
        for move in possible_moves:
            game_copy = self.clone_game(game)
            try:
                if len(move) == 2:
                    if game_copy.is_valid_placement(*move):
                        game_copy.place_checker(*move)
                        priority = self.quick_evaluate(game_copy, self.player)
                        move_priorities[move] = priority
                else:
                    if game_copy.is_valid_move(*move):
                        game_copy.move_checker(*move)
                        priority = self.quick_evaluate(game_copy, self.player)
                        move_priorities[move] = priority
            except:
                continue

        # Sort moves by priority
        sorted_moves = sorted(move_priorities.keys(), 
                            key=lambda m: move_priorities[m], 
                            reverse=True)[:10]  # Only consider top 10 moves
        
        while time.time() < end_time:
            # Select move using UCB1
            total_visits = sum(visits for _, visits in move_stats.values()) + 1
            
            # Try untried moves first
            selected_move = None
            for move in sorted_moves:
                if move not in move_stats:
                    selected_move = move
                    break
            
            # If all moves tried, use UCB1
            if selected_move is None:
                best_ucb = float('-inf')
                for move in sorted_moves:
                    total_score, visits = move_stats.get(move, (0, 0))
                    if visits > 0:
                        exploitation = total_score / visits
                        exploration = exploration_constant * math.sqrt(math.log(total_visits) / visits)
                        ucb = exploitation + exploration
                        if ucb > best_ucb:
                            best_ucb = ucb
                            selected_move = move
            
            if selected_move is None:
                selected_move = random.choice(sorted_moves)
                
            # Simulate game
            score = self.light_simulation(game, selected_move)
            
            # Update statistics
            current_score, visits = move_stats.get(selected_move, (0, 0))
            move_stats[selected_move] = (current_score + score, visits + 1)
        
        # Select best move based on visits
        best_move = None
        most_visits = -1
        
        for move in sorted_moves:
            _, visits = move_stats.get(move, (0, 0))
            if visits > most_visits:
                most_visits = visits
                best_move = move
        
        return best_move if best_move else random.choice(sorted_moves)

    def quick_evaluate(self, game, player):
        """Fast position evaluation"""
        score = 0
        
        # Check for win
        winner = game.check_winner()
        if winner == player:
            return 1000
        elif winner == -player:
            return -1000
            
        # Count pieces and alignments
        for r in range(8):
            for c in range(8):
                if game.board[r][c] == player:
                    score += 1
                    # Check horizontal and vertical alignments
                    if c < 7 and game.board[r][c+1] == player:
                        score += 5
                    if r < 7 and game.board[r+1][c] == player:
                        score += 5
                        
        return score

    def light_simulation(self, game, first_move):
        """Lightweight game simulation"""
        sim_game = self.clone_game(game)
        try:
            # Make first move
            if len(first_move) == 2:
                sim_game.place_checker(*first_move)
            else:
                sim_game.move_checker(*first_move)
                
            sim_game.current_player *= -1
            moves_left = 20  # Reduced simulation length
            
            while moves_left > 0:
                winner = sim_game.check_winner()
                if winner != 0:
                    return 1.0 if winner == self.player else -1.0
                    
                moves = self.get_possible_moves(sim_game)
                if not moves:
                    break
                    
                # Simple random policy with basic pruning
                valid_moves = []
                for move in moves[:10]:  # Only consider first 10 moves
                    try:
                        if len(move) == 2:
                            if sim_game.is_valid_placement(*move):
                                valid_moves.append(move)
                        else:
                            if sim_game.is_valid_move(*move):
                                valid_moves.append(move)
                    except:
                        continue
                
                if not valid_moves:
                    break
                    
                move = random.choice(valid_moves)
                if len(move) == 2:
                    sim_game.place_checker(*move)
                else:
                    sim_game.move_checker(*move)
                    
                sim_game.current_player *= -1
                moves_left -= 1
            
            return self.quick_evaluate(sim_game, self.player) / 1000.0
            
        except:
            return 0.0
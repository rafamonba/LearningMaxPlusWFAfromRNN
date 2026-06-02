import torch
import numpy as np
import itertools
import random
import heapq
from torch.nn.utils.rnn import pack_padded_sequence

class TrainedRNNOracle:
    def __init__(self, rnn_model, vocab_dict, tol=0.5, rounding_step=1):
        """
        Wraps the PyTorch RNN to act as a unified L* Oracle.
        Supports EQ resolution via Hybrid search and Max-Plus Abstraction.
        """
        self.model = rnn_model
        self.vocab_dict = vocab_dict
        self.alphabet = list(vocab_dict.keys())
        self.rounding_step = rounding_step
        self.tol = tol
        self.ZERO = float('-inf')
        
        # Global memory to prevent infinite loops in Equivalence Queries
        self.seen_counterexamples = set()
        
        # Safe device handling (CPU/GPU)
        self.device = next(self.model.parameters()).device
        
        # Evaluation mode to prevent gradient computation (faster and safer)
        self.model.eval()

    # =========================================================================
    # CORE ORACLE METHODS (MEMBERSHIP QUERIES)
    # =========================================================================
    
    def calculate_weight(self, word: str) -> float:
        """
        Resolves Membership Queries (MQ).
        Passes the string to the RNN and returns the predicted cost (rounded).
        """
        if not word:
            return 0.0 
            
        indices = [self.vocab_dict[char] for char in word]
        tensor_in = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        tensor_len = torch.tensor([len(word)], dtype=torch.long).cpu()
        
        with torch.no_grad():
            cost = self.model(tensor_in, tensor_len).item()
            
        if self.rounding_step == 0:
            return float(cost)
        else:
            return float(round(cost / self.rounding_step) * self.rounding_step)

    def _verify_counterexample(self, word: str, hypothesis_wfa) -> bool:
        """
        Verifies if 'word' is a valid counterexample by applying XOR logic 
        for structural impossibilities (-inf) and continuous distance tolerance.
        Returns True if it IS a counterexample.
        """
        if word in self.seen_counterexamples:
            return False
            
        hyp_val = hypothesis_wfa.classify_word(word)
        true_val = self.calculate_weight(word) 
        
        # Equivalent if both are -inf, or if both are finite and their difference is strictly less than tol
        is_equivalent = (hyp_val == self.ZERO and true_val == self.ZERO) or \
                        (hyp_val != self.ZERO and true_val != self.ZERO and abs(hyp_val - true_val) < self.tol)
        
        if not is_equivalent:
            self.seen_counterexamples.add(word)
            print(f"      -> Failure detail for '{word}': RNN={true_val:.2f}, WFA={hyp_val:.2f}")
            return True
            
        return False

    # =========================================================================
    # EQUIVALENCE QUERY DISPATCHER
    # =========================================================================

    def equivalence_query(self, hypothesis_wfa, method="hybrid", **kwargs):
        """
        Main entry point to resolve an Equivalence Query.
        Delegates execution to the requested strategy.
        """
        if method == "hybrid":
            return self._eq_hybrid(hypothesis_wfa, **kwargs)
        elif method == "abstraction":
            return self._eq_abstraction(hypothesis_wfa, **kwargs)
        else:
            raise ValueError(f"Unknown Equivalence Query method: {method}")

    # =========================================================================
    # 1. HYBRID EXHAUSTIVE-RANDOM SEARCH
    # =========================================================================

    def _eq_hybrid(self, hypothesis_wfa, exhaustive_len=6, random_max_len=25, num_random=2000):
        # --- PHASE 1: EXHAUSTIVE SEARCH ---
        print(f"  [EQ-Hybrid] Phase 1: Exhaustive search up to length {exhaustive_len}...")
        for length in range(exhaustive_len + 1):
            for p in itertools.product(self.alphabet, repeat=length):
                word = "".join(p)
                
                if self._verify_counterexample(word, hypothesis_wfa):
                    print(f"\n[!] Counterexample found (Exhaustive): '{word}'")
                    return word

        # --- PHASE 2: RANDOM SEARCH ---
        if random_max_len > exhaustive_len and num_random > 0:
            print(f"  [EQ-Hybrid] Phase 2: Testing {num_random} random words (length {exhaustive_len + 1} to {random_max_len})...")
            for _ in range(num_random):
                random_length = random.randint(exhaustive_len + 1, random_max_len)
                word = "".join(random.choices(self.alphabet, k=random_length))
                
                if self._verify_counterexample(word, hypothesis_wfa):
                    print(f"\n[!] Counterexample found (Random): '{word}'")
                    return word

        print("  [EQ-Hybrid] No counterexamples found. Hypothesis accepted!")
        return None

    # =========================================================================
    # 2. ABSTRACTION-BASED SEARCH (MAX-PLUS)
    # =========================================================================

    def get_rnn_state(self, word: str) -> np.ndarray:
        """
        Extracts the hidden state of the RNN after processing 'word'.
        """
        if not word:
            # Default initial state (PyTorch initializes to 0 if h_0 is not provided)
            return np.zeros(self.model.rnn.hidden_size)
            
        indices = [self.vocab_dict[char] for char in word]
        tensor_in = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
        tensor_len = torch.tensor([len(word)], dtype=torch.long).cpu()
        
        with torch.no_grad():
            emb = self.model.embedding(tensor_in)
            packed_emb = pack_padded_sequence(emb, tensor_len, batch_first=True, enforce_sorted=False)
            _, h_n = self.model.rnn(packed_emb)
            
            # h_n has shape (num_layers, batch, hidden_size). We take the last layer and batch 0.
            return h_n[-1, 0, :].cpu().numpy()

    def maxplus_regression_safe(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes the matrix M = Y / X (Right residuation in Max-Plus)
        X: RNN States (hidden_dim, N)
        Y: WFA Configurations (|Q|, N)
        """
        Y_expanded = Y[:, np.newaxis, :]
        X_expanded = X[np.newaxis, :, :]
        
        with np.errstate(invalid='ignore'):
            diff = Y_expanded - X_expanded
        
        # Infinity rules according to Max-Plus algebra
        mask_neg_inf = np.logical_and(Y_expanded == self.ZERO, X_expanded == self.ZERO)
        mask_pos_inf = np.logical_and(Y_expanded == np.inf, X_expanded == np.inf)
        
        diff[mask_neg_inf] = np.inf
        diff[mask_pos_inf] = np.inf
        
        return np.min(diff, axis=2)

    def predict_wfa_state(self, M: np.ndarray, rnn_state: np.ndarray) -> np.ndarray:
        """
        Computes M (x) rnn_state to project the RNN state into the WFA space.
        """
        sums = M + rnn_state[np.newaxis, :]
        np.nan_to_num(sums, copy=False, nan=self.ZERO)
        return np.max(sums, axis=1)

    def maxplus_linf_dist(self, x: np.ndarray, y: np.ndarray, zero_sub=-10000.0) -> float:
        """
        Computes the L_infinity distance adapted for Max-Plus.
        Replaces -inf with a low baseline ('zero_sub') to avoid breaking the metric.
        """
        x_safe = np.nan_to_num(x, neginf=zero_sub)
        y_safe = np.nan_to_num(y, neginf=zero_sub)
        return float(np.max(np.abs(x_safe - y_safe)))

    def is_consistent(self, h: str, rnn_h: np.ndarray, visited_words: list, 
                      rnn_states_dict: dict, wfa_states_dict: dict, M_matrix: np.ndarray, state_tol: float) -> bool:
        """
        Implements the CONSISTENT? function.
        Evaluates if the current regression model (M_matrix) is locally consistent 
        with the new word 'h' using the Max-Plus adapted L_infinity norm.
        """
        if M_matrix is None:
            return False
            
        p_h = self.predict_wfa_state(M_matrix, rnn_h)
        
        for h_prime in visited_words:
            wfa_h_prime = wfa_states_dict[h_prime]
            rnn_h_prime = rnn_states_dict[h_prime]
            
            p_h_prime = self.predict_wfa_state(M_matrix, rnn_h_prime)
            
            # Condition 1: The model fails on h' 
            error_h_prime = self.maxplus_linf_dist(wfa_h_prime, p_h_prime)
            model_fails_on_h_prime = error_h_prime >= state_tol
            
            # Condition 2: h' is spatially similar to h according to the model
            similarity_h_prime_h = self.maxplus_linf_dist(p_h_prime, p_h)
            h_prime_is_similar_to_h = similarity_h_prime_h < state_tol
            
            # If the model fails in a neighborhood highly similar to the current one, it is NOT consistent
            if model_fails_on_h_prime and h_prime_is_similar_to_h:
                return False
                
        return True

    def _eq_abstraction(self, hypothesis_wfa, max_eq_length=6, M_threshold=5, state_tol=1.0):
        """
        Resolves EQ using Max-Plus algebraic regression and Best-First Search.
        """
        print(f"  [EQ-Abstraction] Starting Max-Plus regression-guided search (Max Len: {max_eq_length})...")
        
        queue = []
        heap_counter = itertools.count() 
        heapq.heappush(queue, (0.0, next(heap_counter), ""))
        
        visited_words = []
        rnn_states_dict = {}  
        wfa_states_dict = {}  
        M_matrix = None        

        while queue:
            neg_pr, _, h = heapq.heappop(queue)

            # Pruning heuristic
            if len(h) > max_eq_length:
                continue
                
            # 1. Check counterexample using the unified validator
            if self._verify_counterexample(h, hypothesis_wfa):
                print(f"\n[!] Counterexample found (Abstraction): '{h}'")
                return h

            # 2. Get states of the new word
            rnn_h = self.get_rnn_state(h)
            wfa_h = hypothesis_wfa.calc_states(h)
            
            # 3. CONSISTENT? function (Decides whether to refine the regression)
            is_ok = self.is_consistent(h, rnn_h, visited_words, rnn_states_dict, wfa_states_dict, M_matrix, state_tol)
            
            if not is_ok:
                all_rnn = list(rnn_states_dict.values()) + [rnn_h]
                all_wfa = list(wfa_states_dict.values()) + [wfa_h]
                
                X = np.stack(all_rnn, axis=1)
                Y = np.stack(all_wfa, axis=1)
                M_matrix = self.maxplus_regression_safe(X, Y)
                
            # 4. Confirm visit and save states
            visited_words.append(h)
            rnn_states_dict[h] = rnn_h
            wfa_states_dict[h] = wfa_h

            # 5. Calculate neighborhood concentration (#vn) using L_infinity
            p_h = self.predict_wfa_state(M_matrix, rnn_h)
            distances = []
            vn = 0 
            
            for h_prime in visited_words[:-1]: 
                p_h_prime = self.predict_wfa_state(M_matrix, rnn_states_dict[h_prime])
                dist = self.maxplus_linf_dist(p_h, p_h_prime)
                distances.append(dist)
                
                if dist < state_tol:
                    vn += 1

            # 6. Guided expansion if the region is sparsely explored
            if vn <= M_threshold:
                pr = min(distances) if distances else (state_tol + 1.0)
                
                for a in self.alphabet:
                    new_word = h + a
                    if new_word not in self.seen_counterexamples and new_word not in visited_words:
                        heapq.heappush(queue, (-pr, next(heap_counter), new_word))

        print("  [EQ-Abstraction] No counterexamples found. Equivalent Hypothesis!")
        return None
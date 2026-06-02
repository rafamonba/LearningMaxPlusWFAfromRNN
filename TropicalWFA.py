from typing import Dict, List
import numpy as np
import itertools
import random

class TropicalWFA:
    def __init__(self,
                 alphabet: List[str],
                 q0: np.ndarray,
                 final: np.ndarray,
                 delta: Dict[str, np.ndarray]) -> None:
        """
        Weighted Finite Automaton in the Tropical (Max-Plus) Semiring.
        :param alphabet: List of alphabet symbols (e.g., ['W', 'O', 'F'])
        :param q0: Vector of initial weights (1D array)
        :param final: Vector of final weights (1D array)
        :param delta: Dictionary containing the transition matrices for each symbol
        """
        self.alphabet = alphabet
        
        # Ensure 1D arrays (flattened) to facilitate broadcasting
        self.q0 = q0.flatten() 
        self.final = final.flatten()
        self.delta = delta
        self.ZERO = float('-inf')

        # Dimensionality safety checks
        assert len(self.alphabet) > 0
        n = self.q0.size
        assert n == self.final.size
        for k, v in self.delta.items():
            assert k in self.alphabet
            assert v.shape == (n, n), f"The matrix for {k} must be {n}x{n}"

    def classify_word(self, word: str) -> float:
        """
        Evaluates a sequence of actions (word) and returns the total weight/benefit.
        Equivalent to: q0 (x) Delta_w1 (x) ... (x) Delta_wn (x) final
        """
        state = self.calc_states(word)
        return self.calc_result(state)

    def calc_states(self, word: str) -> np.ndarray:
        """
        Calculates the state vector after processing the entire word.
        """
        q = self.q0
        for a in word:
            q = self.calc_next(q, a)
        return q

    def calc_next(self, state: np.ndarray, a: str) -> np.ndarray:
        """
        Tropical Vector-Matrix Multiplication: state (x) delta[a]
        Mathematically: q_next[j] = max_i (state[i] + delta[a][i, j])
        """
        # 1. Standard NumPy addition
        sums = state[:, np.newaxis] + self.delta[a]
        
        # 2. Infinity rule correction: If there is a NaN (from -inf + inf), 
        # force it to the tropical zero (-inf)
        np.nan_to_num(sums, copy=False, nan=self.ZERO)
        
        # 3. Take the maximum along columns
        return np.max(sums, axis=0)

    def calc_result(self, state: np.ndarray) -> float:
        """
        Tropical Dot Product: state (x) final
        Mathematically: max_i (state[i] + final[i])
        """
        # 1. Standard addition
        final_sums = state + self.final
        
        # 2. Correct NaNs by forcing them to -inf
        np.nan_to_num(final_sums, copy=False, nan=self.ZERO)
        
        # 3. Return the maximum value
        return float(np.max(final_sums))

    def show_wfa(self) -> str:
        """
        Prints the formal definition of the Automaton.
        """
        s = "=== Tropical Weighted Automaton ===\n"
        s += f"q0 (Initial Weights): {self.q0}\n"
        s += f"final (Final Weights): {self.final}\n"
        s += "--- Transition Matrices ---\n"
        for a in self.alphabet:
            s += f"Delta[{a}]:\n{self.delta[a]}\n"
        return s

    def prune_weights(self, threshold: float = -50.0) -> None:
        """
        Replaces all weights strictly less than `threshold` with `self.ZERO`.
        """
        # 1. Modify initial weights
        self.q0[self.q0 < threshold] = self.ZERO
        
        # 2. Modify final weights
        self.final[self.final < threshold] = self.ZERO
        
        # 3. Modify transition matrices
        for a in self.delta:
            self.delta[a][self.delta[a] < threshold] = self.ZERO  

    def push_weights_to_positive(self) -> None:
        """
        Applies the Weight Pushing (Reparameterization) algorithm using Bellman-Ford.
        It pushes negative transition weights towards the final states, ensuring
        that all transition matrices are >= 0 without altering the global weight 
        of any word.
        """
        n = self.q0.size
        # Initialize the potential vector P to 0
        P = np.zeros(n) 

        # 1. Bellman-Ford: Find the maximum pushing bound (shortest path in min-plus)
        # We want P[j] <= P[i] + delta[a][i,j] to ensure the new weight is >= 0
        for _ in range(n - 1):
            P_next = np.copy(P)
            for a in self.alphabet:
                # Only operate on real transitions (not -inf)
                valid_mask = self.delta[a] != self.ZERO
                
                # Iterate over the matrix to update potentials
                for i in range(n):
                    for j in range(n):
                        if valid_mask[i, j]:
                            # Bellman Equation: P[j] = min(P[j], P[i] + W[i,j])
                            val = P[i] + self.delta[a][i, j]
                            if val < P_next[j]:
                                P_next[j] = val
            P = P_next

        # 2. Reparameterization of Transition Matrices
        # W'_a[i,j] = W_a[i,j] + P[i] - P[j]
        for a in self.alphabet:
            valid_mask = self.delta[a] != self.ZERO
            # Create a shift matrix using broadcasting
            shift_matrix = P[:, np.newaxis] - P[np.newaxis, :]
            # Apply the shift only to valid transitions
            self.delta[a][valid_mask] += shift_matrix[valid_mask]

        # 3. Reparameterization of Initial Weights
        # q0'[i] = q0[i] - P[i]
        valid_q0 = self.q0 != self.ZERO
        self.q0[valid_q0] -= P[valid_q0]

        # 4. Reparameterization of Final Weights
        # final'[i] = final[i] + P[i]
        valid_final = self.final != self.ZERO
        self.final[valid_final] += P[valid_final]
        
        print("\n[+] Weight Pushing applied: Negative weights pushed towards the endpoints.")    

    def prune_by_viterbi(self, sample_words: List[str]) -> None:
        """
        Prunes 'junk' transitions from a dense automaton and removes disconnected states.
        Traces the winning path (Viterbi) for a set of sample words.
        """
        k = len(self.q0)
        
        # Boolean masks to track what is actually used
        used_q0 = np.zeros(k, dtype=bool)
        used_final = np.zeros(k, dtype=bool)
        used_delta = {a: np.zeros((k, k), dtype=bool) for a in self.alphabet}
        
        print(f"Starting Viterbi pruning with {len(sample_words)} sequences...")
        
        for word in sample_words:
            if not word:
                continue
                
            n = len(word)
            dp = np.full((n + 1, k), self.ZERO)
            backpointers = np.zeros((n, k), dtype=int)
            
            dp[0] = self.q0
            
            # Forward pass (Viterbi)
            for t, a in enumerate(word):
                for j in range(k):
                    vals = dp[t] + self.delta[a][:, j]
                    best_i = np.argmax(vals)
                    dp[t+1, j] = vals[best_i]
                    backpointers[t, j] = best_i
                    
            final_vals = dp[n] + self.final
            best_final_state = np.argmax(final_vals)
            
            # Backward pass 
            curr_state = best_final_state
            used_final[curr_state] = True
            
            for t in range(n - 1, -1, -1):
                a = word[t]
                prev_state = backpointers[t, curr_state]
                
                used_delta[a][prev_state, curr_state] = True
                curr_state = prev_state
                
            used_q0[curr_state] = True

        # ==========================================
        # PHASE 1: Transition Cleanup
        # ==========================================
        count_pruned = 0
        self.q0[~used_q0] = self.ZERO
        self.final[~used_final] = self.ZERO
        
        for a in self.alphabet:
            mask = ~used_delta[a]
            count_pruned += np.sum(mask)
            self.delta[a][mask] = self.ZERO
            
        print(f"[+] Transition pruning: Removed {count_pruned} useless edges.")

        # ==========================================
        # PHASE 2: Orphan State Destruction
        # ==========================================
        # A state is useful if it has been initial, final, source, or target of any edge
        state_is_used = used_q0 | used_final
        for a in self.alphabet:
            state_is_used |= used_delta[a].any(axis=0) # Check if it was a target (columns)
            state_is_used |= used_delta[a].any(axis=1) # Check if it was a source (rows)

        # Get actual indices of surviving states
        surviving_states = np.where(state_is_used)[0]
        k_new = len(surviving_states)
        k_removed = k - k_new

        if k_removed > 0:
            print(f"[+] State cleanup: Removed {k_removed} phantom states. (From {k} to {k_new})")
            
            # Reduce initial and final vectors
            self.q0 = self.q0[surviving_states]
            self.final = self.final[surviving_states]
            
            # Reduce all matrices using np.ix_ to take only the useful intersection
            for a in self.alphabet:
                self.delta[a] = self.delta[a][np.ix_(surviving_states, surviving_states)]
        else:
            print(f"[+] State cleanup: All {k} states were useful. None removed.")

    def normalize_global_bias(self, max_val=True) -> None:
        """
        Removes inflated initial weights by shifting the constant towards 
        the final weights. Maintains strict equivalence.
        """
        valid_q0 = self.q0 != self.ZERO
        if not np.any(valid_q0):
            return
            
        # Find the max/min value the automaton starts with
        if max_val:
            shift = np.max(self.q0[valid_q0])
        else:
            shift = np.min(self.q0[valid_q0])
            
        # Subtract from the entrance
        self.q0[valid_q0] -= shift
        
        # Add to the exit (compensation)
        valid_final = self.final != self.ZERO
        self.final[valid_final] += shift
        
        print(f"[+] Bias Normalized: Global weight shifted by {-shift:.2f}.")

    def trim_dead_states(self) -> None:
        """
        Removes (forces to -inf) states that have no outgoing transitions
        and whose final weight is worse than the main states.
        """
        k = self.q0.size
        states_to_remove = []
        
        for i in range(k):
            # Check for any valid outgoing edge
            has_outgoing = False
            for a in self.alphabet:
                if np.any(self.delta[a][i, :] != self.ZERO):
                    has_outgoing = True
                    break
            
            # If no outgoing edges and it's not the only existing state
            if not has_outgoing and k > 1:
                states_to_remove.append(i)
                
        for i in states_to_remove:
            self.q0[i] = self.ZERO
            self.final[i] = self.ZERO
            for a in self.alphabet:
                self.delta[a][:, i] = self.ZERO # Destroy incoming edges
                
        if states_to_remove:
            print(f"[+] Structural Pruning: Removed sink states: {states_to_remove}")            

    def push_final_weights_to_zero(self) -> None:
        """
        Pushes final weights towards transition matrices and initial weights.
        Ensures all valid final states have a cost of 0.0, preserving strict 
        equivalence via telescoping sums.
        """
        valid_final = self.final != self.ZERO
        if not np.any(valid_final):
            return  # Nothing to push
            
        # 1. The potential P is simply the current final weight.
        # For non-final states (-inf), their pushing potential is 0.
        P = np.zeros_like(self.final, dtype=float)
        P[valid_final] = self.final[valid_final]
        
        # 2. Reparameterize transition matrices: W'_a[i,j] = W_a[i,j] + P[j] - P[i]
        for a in self.alphabet:
            valid_mask = self.delta[a] != self.ZERO
            # Use NumPy broadcasting to create the shift matrix
            shift_matrix = P[np.newaxis, :] - P[:, np.newaxis]
            self.delta[a][valid_mask] += shift_matrix[valid_mask]
            
        # 3. Reparameterize initial weights: q0'[i] = q0[i] + P[i]
        valid_q0 = self.q0 != self.ZERO
        self.q0[valid_q0] += P[valid_q0]
        
        # 4. Reparameterize final weights: final'[i] = final[i] - P[i] -> Becomes 0!
        self.final[valid_final] -= P[valid_final]
        
        print("[+] Backward Weight Pushing: Final weights absorbed to 0.0.")

    def check_twins_property(self, tol: float = 1e-4) -> bool:
        """
        Checks the Twins Property. Mathematical prerequisite to guarantee 
        that the non-deterministic WFA can be determinized without infinite loops.
        """
        k = self.q0.size
        
        # 1. Find pairs of "twin" states (reachable by the same prefix)
        accessible_pairs = set()
        queue = []
        
        # Initial pairs: any pair of valid states at the start
        valid_q0 = np.where(self.q0 != self.ZERO)[0]
        for i in valid_q0:
            for j in valid_q0:
                pair = (i, j)
                accessible_pairs.add(pair)
                queue.append(pair)
                
        edges = []
        
        # BFS to build the graph of reachable pairs and extract its edges
        while queue:
            u1, u2 = queue.pop(0)
            
            for a in self.alphabet:
                # Valid outgoing transitions for u1 and u2
                out1 = np.where(self.delta[a][u1, :] != self.ZERO)[0]
                out2 = np.where(self.delta[a][u2, :] != self.ZERO)[0]
                
                for v1 in out1:
                    for v2 in out2:
                        # The "weight" of this cross edge is the difference in costs
                        w1 = self.delta[a][u1, v1]
                        w2 = self.delta[a][u2, v2]
                        weight_diff = w1 - w2
                        
                        edges.append(((u1, u2), (v1, v2), weight_diff))
                        
                        new_pair = (v1, v2)
                        if new_pair not in accessible_pairs:
                            accessible_pairs.add(new_pair)
                            queue.append(new_pair)
                            
        # 2. Check if all cycles have an accumulated weight of 0 
        # We use Bellman-Ford initializing all nodes at distance 0.
        
        # --- Check for negative divergent cycles ---
        dist = {pair: 0.0 for pair in accessible_pairs}
        num_nodes = len(accessible_pairs)
        
        for _ in range(num_nodes):
            for u, v, w in edges:
                if dist[u] + w < dist[v] - tol:
                    dist[v] = dist[u] + w
                    
        for u, v, w in edges:
            if dist[u] + w < dist[v] - tol:
                print(f"[!] Property Failed: Pair {u} and {v} accumulate infinite delay.")
                return False
                
        # --- Check for positive divergent cycles (inverting the weight) ---
        dist = {pair: 0.0 for pair in accessible_pairs}
        for _ in range(num_nodes):
            for u, v, w in edges:
                if dist[u] - w < dist[v] - tol:
                    dist[v] = dist[u] - w
                    
        for u, v, w in edges:
            if dist[u] - w < dist[v] - tol:
                print(f"[!] Property Failed: Pair {u} and {v} accumulate infinite delay.")
                return False
                
        print("[OK] The Automaton satisfies the Twins Property. It is 100% determinizable!")
        return True        

    def get_critical_path(self) -> dict:
        """
        Finds the Maximum Mean Weight Cycle (Critical Path) in the WFA 
        using Karp's Algorithm adapted for the Max-Plus semiring.
        Returns a dictionary with the maximum mean weight and the critical sequence.
        """
        n = self.q0.size
        if n == 0:
            return {"mean_weight": self.ZERO, "cycle": ""}

        # 1. Condense multi-edges into a single adjacency matrix with max weights
        # W[i, j] stores the heaviest transition from state i to state j
        W = np.full((n, n), self.ZERO)
        symbols = np.empty((n, n), dtype=object)
        
        for a in self.alphabet:
            mask = self.delta[a] != self.ZERO
            better_mask = mask & (self.delta[a] > W)
            W[better_mask] = self.delta[a][better_mask]
            symbols[better_mask] = a

        # 2. Initialize F[k, v]: Max weight of a path of exactly length k ending at v
        F = np.full((n + 1, n), self.ZERO)
        # We implicitly start from a fictitious source node connected to all states
        F[0, :] = 0.0  

        # To reconstruct the sequence, we keep track of the parent states
        parent = np.full((n + 1, n), -1, dtype=int)

        # 3. Compute F[k, v] using Dynamic Programming
        for k in range(1, n + 1):
            for v in range(n):
                vals = F[k-1, :] + W[:, v]
                np.nan_to_num(vals, copy=False, nan=self.ZERO)
                
                best_u = np.argmax(vals)
                max_val = vals[best_u]
                
                if max_val != self.ZERO:
                    F[k, v] = max_val
                    parent[k, v] = best_u

        # 4. Calculate Maximum Mean Weight
        # Formula: max_v { min_{0 <= k < n} [ (F[n, v] - F[k, v]) / (n - k) ] }
        max_mean_weight = self.ZERO
        best_v = -1

        for v in range(n):
            if F[n, v] == self.ZERO:
                continue
                
            min_k_val = float('inf')
            for k in range(n):
                if F[k, v] != self.ZERO:
                    avg = (F[n, v] - F[k, v]) / (n - k)
                    if avg < min_k_val:
                        min_k_val = avg
                        
            if min_k_val > max_mean_weight and min_k_val != float('inf'):
                max_mean_weight = min_k_val
                best_v = v

        if best_v == -1:
            return {"mean_weight": self.ZERO, "cycle": ""}

        # 5. Extract the critical cycle by backtracking from best_v
        path = []
        curr = best_v
        for k in range(n, 0, -1):
            p = parent[k, curr]
            path.append((p, curr, symbols[p, curr]))
            curr = p
            
        path.reverse()
        
        # 6. Find the exact cycle using the Pigeonhole Principle
        nodes = [path[0][0]] + [edge[1] for edge in path]
        visited = {}
        start_idx = -1
        end_idx = -1
        
        for i, node in enumerate(nodes):
            if node in visited:
                start_idx = visited[node]
                end_idx = i
                break
            visited[node] = i
            
        if start_idx != -1:
            cycle_edges = path[start_idx:end_idx]
            critical_sequence = "".join([sym for u, v, sym in cycle_edges])
        else:
            # Fallback sequence
            critical_sequence = "".join([sym for u, v, sym in path])

        return {
            "mean_weight": float(max_mean_weight),
            "cycle": critical_sequence
        }
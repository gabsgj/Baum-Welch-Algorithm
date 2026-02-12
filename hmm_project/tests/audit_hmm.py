
import numpy as np
from hmm_core.inference.components.alpha import compute_alpha
from hmm_core.inference.components.beta import compute_beta
from hmm_core.inference.scaling import compute_log_likelihood
from hmm_core.inference.components.gamma import compute_gamma
from hmm_core.inference.components.xi import compute_xi

def audit_hmm():
    # Setup from Reference
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], 
                  [0.4, 0.6]])
    B = np.array([[0.1, 0.9], 
                  [0.6, 0.4]])
    
    # Obs: W, H. (Index 0, 1)
    observations = np.array([0, 1]) # W, H
    
    # Official Results
    expected_alpha1 = np.array([0.06, 0.24])
    expected_alpha2 = np.array([0.1242, 0.0648]) # Unscaled alpha?
    # User says "alpha1(R) = 0.06"
    # User says "alpha2(R) = 0.1242"
    
    print("--- Running Audit ---")
    
    # Run Scaled Forward
    alpha, c = compute_alpha(A, B, pi, observations)
    
    print("\nScaled Alpha:\n", alpha)
    print("Scaling Factors (c):\n", c)
    
    # Check if alpha is actually scaled
    # Scaled alpha_t sums to 1?
    print("Alpha Sums:", alpha.sum(axis=1))
    
    # User's "alpha" values seem UNSCALED.
    # Ref: alpha1(R)=0.06. 
    # My Pi[0]*B[0,0] = 0.6 * 0.1 = 0.06. Yes.
    # Scaled alpha would be 0.06 / (0.06+0.24) = 0.06/0.3 = 0.2.
    
    # The user asks me to "Implement forward exactly as defined in the PDF."
    # AND "Implement backward exactly as defined."
    # BUT Step 2 says "Verify Scaling Implementation... Ensure alpha is scaled".
    # This implies the CODE uses scaling, but the REFERENCE numbers might be UNSCALED or Scaled.
    # Let's check the reference numbers consistency.
    # alpha1: Sum = 0.06+0.24 = 0.30. P(O1).
    # alpha2: Sum = 0.1242+0.0648 = 0.189. P(O1, O2).
    # So the reference numbers are UNSCALED.
    
    # However, my codebase implements SCALED Forward.
    # So I need to convert my scaled alpha back to unscaled to compare.
    # Unscaled Alpha_t = Scaled Alpha_t / product(c_1...c_t).
    # Or rather: Scaled Alpha_t = Unscaled Alpha_t * (prod c_k).
    # So Unscaled = Scaled / (prod c_k).
    
    # Let's verify.
    prod_c = np.cumprod(c) # c[0] cancels sum_alpha_0? 
    # c_t = 1 / sum(unscaled_alpha_t).
    # So Unscaled Alpha_t = Scaled Alpha_t / (c_t * c_{t-1} ...) ?
    # Wait, c_t scales the step.
    # alpha_scaled_t = alpha_unscaled_t * C_t * ... * C_1 ? No.
    # alpha_scaled_t = alpha_hat_t.
    # alpha_hat_t = alpha_hat_{t-1} * A * B * c_t.
    
    unscaled_reconstructed = np.zeros_like(alpha)
    for t in range(len(observations)):
        scale = np.prod(c[:t+1]) # Product up to t?
        # alpha_unscaled_t = alpha_scaled_t / scale? 
        # Actually P(O_1...O_t) = 1 / product(c_1...c_t).
        # And sum(alpha_unscaled) = P(O_1...O_t).
        # sum(alpha_scaled) = 1.
        # So conversion factor matches.
        
        factor = 1.0 / np.prod(c[:t+1]) # This is P(O...t)
        # Wait, product(c) is 1/P. So 1/product(c) is P.
        # So Unscaled = Scaled * (1/product(c)) ? No.
        # Scaled = Unscaled * product(c).
        # So Unscaled = Scaled / product(c).
        unscaled_reconstructed[t] = alpha[t] / np.prod(c[:t+1])
        
    print("\nReconstructed Unscaled Alpha:\n", unscaled_reconstructed)
    
    # Compare with expected
    print("Expected Alpha 1:", expected_alpha1)
    print("Expected Alpha 2:", expected_alpha2)
    
    # Backward
    beta = compute_beta(A, B, observations, c)
    print("\nScaled Beta:\n", beta)
    
    # Reconstruct Unscaled Beta
    # User Beta 2 = [1, 1]. Unscaled initialized to 1.
    # My Beta 2 = c_T * 1. 
    # c_2 corresponds to T=2.
    # scaling_factors[T-1] is c_T.
    # My beta matches scaled definition.
    
    # Unscaled Beta_t = Scaled Beta_t / (product c_t...c_T)?
    # Scaling logic is tricky for Beta.
    # Let's just compute Unscaled Beta manually to verify.
    
    beta_unscaled_manual = np.zeros_like(beta)
    T = len(observations)
    beta_unscaled_manual[T-1] = 1.0
    for t in range(T-2, -1, -1):
        beta_unscaled_manual[t] = A @ (B[:, observations[t+1]] * beta_unscaled_manual[t+1])
        
    print("Manual Unscaled Beta:\n", beta_unscaled_manual)
    
    # User Expected:
    # Beta2: 1, 1
    # Beta1: 0.75, 0.60
    
    # Likelihood
    ll = compute_log_likelihood(c)
    prob = np.exp(ll)
    print(f"\nLog Likelihood: {ll}")
    print(f"P(O|lambda): {prob}")
    print(f"Expected P(O): 0.189")
    
    if np.abs(prob - 0.189) < 1e-4:
        print("Likelihood MATCHES.")
    else:
        print("Likelihood MISMATCH.")
        
    # Gamma
    # User: Gamma1(R) = 0.238.
    # gamma = alpha * beta * c? No.
    # Scaled Forward Backward: gamma_t = alpha_scaled * beta_scaled / c_t ??
    # Actually alpha_scaled * beta_scaled = alpha_unscaled * Prod(c_1..t) * beta_unscaled * Prod(c_t..T).
    # = (alpha_un * beta_un) * Prod(c_1..T) * c_t.
    # = P(O, q_t) * (1/P(O)) * c_t.
    # = P(q_t|O) * c_t.
    # So gamma = alpha_scaled * beta_scaled / c_t.
    
    # Wait, let's verify my gamma.py
    
    # audit_hmm()
    
    # Test Long Sequence
    print("\n--- Testing Length 50 ---")
    # Reference Params
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.9], [0.6, 0.4]])
    
    obs_long = np.array([0, 1] * 25) # Length 50
    alpha, c = compute_alpha(A, B, pi, obs_long)
    ll = compute_log_likelihood(c)
    print(f"Log Likelihood (T=50): {ll}")
    print(f"P(O): {np.exp(ll)}")
    print(f"Average c_t: {c.mean()}")
    print(f"Sum log c_t: {np.sum(np.log(c))}")

if __name__ == "__main__":
    audit_hmm()

import numpy as np

# ===== Function for Part (a) =====
def f1(x):
    return np.exp(2 * np.sin(x)) - 2 * x - 1

def f1_prime(x):
    return 2 * np.exp(2 * np.sin(x)) * np.cos(x) - 2

def f1_double_prime(x):
    return 2 * np.exp(2 * np.sin(x)) * (2 * np.cos(x)**2 - np.sin(x))

# ===== Function for Part (c) =====
def f2(x):
    return (8 * x**2) / (3 * x**2 + 1)

def f2_prime(x):
    num = 16 * x * (1 + 3 * x**2) - 48 * x**3
    denom = (3 * x**2 + 1)**2
    return num / denom

# ===== Standard Newton’s Method =====
def newton_method(f, f_prime, x0, iterations, epsilon=1e-10):
    x = x0
    for i in range(iterations):
        denom = f_prime(x)
        if abs(denom) < epsilon:
            print(f"Stopped: derivative too small at iteration {i}, x = {x}")
            break
        x = x - f(x) / denom
    return x

# ===== Modified Newton’s Method (for multiplicity 2) =====
def modified_newton_method(f, f_prime, x0, iterations, epsilon=1e-10):
    x = x0
    for i in range(iterations):
        denom = f_prime(x)
        if abs(denom) < epsilon:
            print(f"Stopped: derivative too small at iteration {i}, x = {x}")
            break
        x = x - 2 * f(x) / denom
    return x

# ======= Part (a): Verify Multiplicity 2 =======
f_at_0 = f1(0)
f_prime_at_0 = f1_prime(0)
f_double_prime_at_0 = f1_double_prime(0)

print("=== Part (a): Root Multiplicity Check ===")
print(f"f(0) = {f_at_0}")
print(f"f'(0) = {f_prime_at_0}")
print(f"f''(0) = {f_double_prime_at_0}")
print("→ Conclusion: x = 0 is a root of multiplicity 2.\n")

# ======= Part (b): f1 using x0 = 0.1 =======
x0_b = 0.1
x9_newton_f1 = newton_method(f1, f1_prime, x0_b, 9)
x9_modified_f1 = modified_newton_method(f1, f1_prime, x0_b, 9)

print("=== Part (b): f(x) = e^{2sin(x)} - 2x - 1 ===")
print(f"x₉ (Newton’s Method) = {x9_newton_f1}")
print(f"x₉ (Modified Newton’s Method) = {x9_modified_f1}\n")

# ======= Part (c): f2 using x0 = 0.15 =======
x0_c = 0.15
x9_newton_f2 = newton_method(f2, f2_prime, x0_c, 9)
x9_modified_f2 = modified_newton_method(f2, f2_prime, x0_c, 9)

print("=== Part (c): f(x) = (8x²)/(3x² + 1) ===")
print(f"x₉ (Newton’s Method) = {x9_newton_f2}")
print(f"x₉ (Modified Newton’s Method) = {x9_modified_f2}")

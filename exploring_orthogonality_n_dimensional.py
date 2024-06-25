import numpy as np

def are_orthogonal(v1, v2):
    dot_product = np.dot(v1, v2)
    return np.isclose(dot_product, 0)

def gram_schmidt(vectors):
    ortho_basis = []
    for v in vectors:
        u = v - sum(np.dot(v, w) / np.dot(w, w) * w for w in ortho_basis)
        if np.linalg.norm(u) > 1e-10:  # Avoid adding a zero vector
            ortho_basis.append(u / np.linalg.norm(u))
    return ortho_basis

def generate_orthogonal_vectors(dim, num_vectors):
    random_vectors = [np.random.rand(dim) for _ in range(num_vectors)]
    orthogonal_vectors = gram_schmidt(random_vectors)
    return orthogonal_vectors

def main():
    # Define example dimensions and number of vectors
    dim = 3
    num_vectors = 3

    # Generate orthogonal vectors
    orthogonal_vectors = generate_orthogonal_vectors(dim, num_vectors)

    # Print generated orthogonal vectors
    print("Generated orthogonal vectors:")
    for i, vec in enumerate(orthogonal_vectors):
        print("Vector {}: {}".format(i+1, vec))

if __name__ == "__main__":
    main()

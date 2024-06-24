import numpy as np

def are_orthogonal(v1, v2):
    """
    Check if two vectors are orthogonal.
    
    Args:
    - v1, v2: numpy arrays representing the vectors.
    
    Returns:
    - bool: True if vectors are orthogonal, False otherwise.
    """
    dot_product = np.dot(v1, v2)
    return np.isclose(dot_product, 0)

def gram_schmidt(vectors):
    """
    Applies the Gram-Schmidt process to a set of vectors to orthogonalize them.
    
    Args:
    - vectors: list of numpy arrays representing the vectors.
    
    Returns:
    - list of numpy arrays: Orthogonalized vectors.
    """
    ortho_basis = []
    for v in vectors:
        u = v - sum(np.dot(v, w) / np.dot(w, w) * w for w in ortho_basis)
        if np.linalg.norm(u) > 1e-10:  # Avoid adding a zero vector
            ortho_basis.append(u / np.linalg.norm(u))
    return ortho_basis

def generate_orthogonal_vectors(dim, num_vectors):
    """
    Generate a set of orthogonal vectors in n-dimensional space using Gram-Schmidt process.
    
    Args:
    - dim: The dimension of the space.
    - num_vectors: The number of orthogonal vectors to generate.
    
    Returns:
    - list of numpy arrays: List of orthogonal vectors.
    """
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



## Ուղղա նկյունությա ն նշա նա կութ յունը
## Գ ծա յին հա նրա հա շիվ.
## 
## Ուղղա նկյունութ յունը վճռորոշ է վեկտ որային տա րա ծությունների ներկա յա ցումը պ ա րզեցնելու հա մա ր:
## Ուղղա նկյուն վեկտ որները հիմք են հա նդիսա նում բա զմա թիվ մա տ րիցա յին ֆա կտ որիզա ցիա ների, ինչպ իսին է QR տա րրա լուծումը:
## Ուղղա նկ յուն պ րոյեկցիա ները օգտ ա գործվում են գծա յին հա մա կա րգերի նվա զա գույն քա ռա կուսիների լուծումներում:
## Օ պ տ իմա լա ցում:

## Օպտիմալացման ժամանակ ուղղանկյուն փոխակերպումները պահպանում են հեռավորությունները և անկյունները, ինչը օգտակար է թվային կայունության համար:
## Գրադիենտ անկումը և այլ օպտիմալացման ալգորիթմները հաճախ հիմնվում են ուղղանկյուն հատկությունների վրա՝ արդյունավետորեն մերձեցման համար:
~
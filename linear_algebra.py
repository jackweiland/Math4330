def absolute(number):
    """ Computes the magnitude

    Computes the square root of the sum of real part squared and the imaginary
    part squared. Then returns the result.

    Args:
        number: an abritray number.
    Returns:
        The magnitude of the input number as a nubmer.
    """
    try:
        result = (number.real ** 2 + number.imag ** 2) ** (1/2)
        return result
    except AttributeError as err:
        print("Error:", err)

def complexConjugate(complex_number):
    """ Compute the complex conjugate of a number

    Sets the result as the real part of the input, then
    subtracts the imaginary part of the input from the result.
    Finally returns result.

    Args:
        complex_number: An arbitrary complex number
    Returns:
        The complex conjugate of the input
    """
    try:
        result = complex_number.real
        result = result - complex_number.imag * 1j
        return result
    except AttributeError as err:
        print("Error:", err)

def vectorAdd(vector_x, vector_y):
    """ Computes the sum of two vectors

    Initializes result to be a zero vector with the same length as
    vector_x. Each element of result is redefined to be the sum of
    the corrisponding elements of vector_x and vector_y. Finally
    returns result.

    Args:
        vector_x: An arbitrary vector of arbitrary length.
                  Represented as a list of numbers.
        vector_y: An arbitrary vector the same length as vector_x.
                  Represented as a list of numbers.
    Returns:
        A list of the same length as vector_x which is the sum
        of two input vectors.
    """
    try:
        if len(vector_x) == len(vector_y):
            result = [0] * len(vector_x)
            for index in range(len(vector_x)):
                result[index] = vector_x[index] + vector_y[index]
            return result
        else:
            print("Error: Input vectors are not of compatible length.")
    except TypeError as err:
        print("Error:", err)

def pNorm(vector, p):
    """ P-norm of a vector

    Result is set to be zero. The absolute value of each element of the
    vector is taken to the power of p, then added to result. After this
    is done to every element, the result is taken to the power of 1/p.
    Finally returns result.

    Args:
        vector: An arbitrary vector of arbitrary length. Represented as
                a list of numbers.
        p: An arbitrary real number greater than or equal to one.

    Returns:
        The p-norm of the given vector as a real number greater than or equal to 0.
    """
    result = 0
    for index in range(len(vector)):
        result += absolute(vector[index]) ** p
    result = result ** (1/p)
    return result

def twoNorm(vector):
    """ Calculates the two norm of a vector

    Initilizes result to be zero. The absolute value of each element of the
    input vector is taken to the power of two and added to result. Then after
    this has been done to every element the result is taken to the power of 1/2.
    Finally returns the result.

    Args:
        vector: An arbitrary vector of arbitrary length. Represented as a
                list of numbers.

    Returns:
        The two norm of the vector. Represented as a real number greater than or equal to 0.
    """
    result = 0
    for index in range(len(vector)):
        result += absolute(vector[index]) ** 2
    result = result ** (1/2)
    return result

def maxNorm(vector):
    """ Calculates the max norm of a vector

    Initilizes result to be the first index of vector. Then for every element of the input vector
    it will compare it to result. If the element is bigger then it will become result.

    Args:
        vector: An arbitrary vector of arbitrary length. Repreesented as a
                list of numbers.

    Returns:
        The max norm of the vector. Represented as a real number greater than or equal to 0.
    """
    result = vector[0]
    for index in range(len(vector)):
        if vector[index] > result:
            result = vector[index]
    return result

def scalarVectorMulti(vector, scalar):
    """ Vector and scalar product

    Initializes result to be a zero vector with the same length as the input
    vector. Each element of the result is redefined to be the product of the
    given scalar and the corrisponding element of the input vector. Finally
    returns the result. Finally returns the result.

    Args:
        vector: An arbitrary vector of arbitrary length. Represented as a
                list of numbers.
        scalar: An arbitrary number.

    Returns:
        The product of the vector and scalar as a vector with the same length
        as the input vector. Represented as a list of numbers.
    """
    result = [0] * len(vector)
    for index in range(len(vector)):
        result[index] = scalar * vector[index]
    return result

def dot(vector_x, vector_y):
    """ Dot product of two vectors

    Initilizes result to be zero. Redefines result to be the sum of the product of
    each elemnt of the conjugate of vector_x and the corrisponding element of
    vector_y. Then returns the result.

    Args:
        vector_x: An arbitrary vector of arbitrary length. Represented as
                  a list of numbers.
        vector_y: An arbitrary vector with the same length as vector_x.
                  Represented as a list of numbers.

    Returns:
        The dot product of the two vectors. Represented as a number.
    """
    result = 0
    for index in range(len(vector_x)):
        result += complexConjugate(vector_x[index]) * vector_y[index]
    return result

def vectorConjugate(vector):
    """ Conjugate of a vector

    Initializes result to be a zero vector with the same length as
    the input vector. Each element of the result vector is redefined to
    be the conjugate of the corrisponding element of the input vector.
    Finally returns the result.

    Args:
        vector: An arbitrary vector of arbitrary length. Represented as
                a list of numbers.

    Returns:
        The conjugate of the input vector with the same length as the input
        vector. Represented as a list of numbers.
    """
    result = [0] * len(vector)
    for index in range(len(vector)):
        result[index] = complexConjugate(vector[index])
    return result

def matrixVectorMulti(matrix, vector):
    """ Matrix and vector product

    Initializes a zero vector with compatible length. For each column vector in
    the matrix, the result is redefined to be the vector sum of the result and the
    product between the column of the matrix and the corrisponding element of the
    vector. Finally returns the result.

    Args:
        matrix: An arbitrary matrix of arbitrary dimensions. Represented as a
                list of column vectors.
        vector: An arbitrary vector of compatible length. Represented as a
                list of numbers.
    Returns:
        The product of the matrix and vector as a vector with a compatible length.
        Represented as a list of numbers.
    """
    result = [0] * len(matrix[0])
    for index in range(len(matrix)):
        temp = scalarVectorMulti(matrix[index], vector[index])
        result =  vectorAdd(result, temp)
    return result

def scalarMatrixMulti(matrix, scalar):
    """ Scalar and matrix product

    Initializes result as a zero vector with length equivalent to the number
    of columns of the matrix. Each element of the result is redefined to be
    the product of the matrix column and the scalar. Finally returns result.

    Args:
        matrix: An arbitrary matrix of arbitrary length. Represented as a list
                of column vectors.
        scalar: An arbitrary number.

    Returns:
        The product of the scalar and the matrix as a matrix with the same
        dimensions as the input matrix. Represented as a list of column vectors.
    """
    result = [0] * len(matrix)
    for index in range(len(matrix)):
        result[index] = scalarVectorMulti(matrix[index], scalar)
    return result

def matrixConjugate(matrix):
    """ Conjugate each elements of a matrix

    Initilizes result to be a zero vector with the length equivelent to
    the number of columns in the input matrix. Each element of result is
    redefined to be the conjugate of the corrisponding columns of the matrix.
    Finally returns the result.

    Args:
        matrix: An arbitrary matrix of arbitrary length. Represented as a
                list of column vectors.
    Returns:
        The conjugate of the input matrix with the same dimensions as the matrix.
        Represented as a list of column vectors.
    """
    result = [0] * len(matrix)
    for index in range(len(matrix)):
        result[index] = vectorConjugate(matrix[index])
    return result

def matrixMulti(matrix_A, matrix_B):
    """ Product of two matices

    Initilizes result to be a zero vector with the same length as
    the number of columns in matrix_B. Each element of result is redefined
    to be the product of matrix_A and the corrisponding column of matrix_B.
    Finally returns the result.

    Args:
        matrix_A: An arbitrary matrix of arbitrary dimensions.
                  Represented as list of column vectors.
        matrix_B: An arbitrary matrix of compatible dimension for
                  matrix_A. Represented as a list of column vectors.

    Returns:
        The matrix that is the product of matrix_A and matrix_B with dimensions
        equivalent to the product of the two vectors. Represented
        as a list of column vectors.
    """
    result = [0] * len(matrix_B)
    for index in range(len(matrix_B)):
        result[index] = matrixVectorMulti(matrix_A, matrix_B[index])
    return result

def conjugateTranspose(matrix):
    """ Conjugate transpose of a matrix

    Initilizes result to be a zero matrix with same dimensions of the transpose
    of the input matrix. Then the input matrix has each of its elements conjugated
    and this is stored in conjugate. Then result is set to be the transpose of the
    conjugate matrix. Finally returns the result.

    Args:
        matrix: An arbitrary matrix of arbitrary dimensions. Represented as a
                list of column vectors.
    Returns:
        The transpose of the input where every value has been conjugated.
    """
    result =  [[]] * len(matrix[0])
    for index in range(len(matrix[0])):
        result[index] = [0] * len(matrix)
    conjugate = matrixConjugate(matrix)
    for index1 in range(len(conjugate)):
        for index2 in range(len(conjugate[0])):
            result[index2][index1] = conjugate[index1][index2]
    return result

def orthoDecomp(ortho_set, vector):
    """ Computes an orthogonal vector.

    This function computes the orthogonal decomposition of vector with
    respect to ortho_set

    Args:
        ortho_set: A list of lists, where each elemnt represents a vector and the
                   list as a whole is orthonormal
        vector: A vector of compativle dimensions to the vetors in ortho_set.
                Represented as a list of numbers

    Returns:
        A vetor which is orthogonal to the vetors in ortho_set with the same
        dimensions as vector. Represented as a list of numbers.
    """
    result = [0] * len(vector)
    for index in range(len(ortho_set)):
        temp0 = dot(ortho_set[index], vector)
        temp1 = scalarVectorMulti(ortho_set[index], -1 * temp0)
        result = vectorAdd(result, temp1)
    return result

def normalize(vector):
    """ Normalizes a vector

    Sets temp to be the two norm of the input vector. For each element in the input vector
    the element will be multiplied by the reciprocal of temp. If temp is 0 then input is invalid.

    Args:
        vector: an arbitrary vector of arbitrary length. Represented as a list of numbers.
    Returns:
        The normalized vector with respect to the two norm with the same length as the input
        vector. Represented as a list of numbers.
    """
    temp = twoNorm(vector)
    if temp == 0:
        print("Error: The zero vector can not be normalized.")
    else:
        result = scalarVectorMulti(vector, 1 / temp)
        return result

def qrFactor(matrix_a):
    """ QR factorization modified Gram-Schdmit

    Initilizes matrix_q, matrix_r, and temp_set. For each index0 from 0 to the
    number of columns of matrix_a - 1: the index0 diagonal of matrix_r is
    set to be the two norm of the index0 element of temp_set, the index0 element
    of matrix_q is set to be the normal vector of the previous diagonal of
    matrix_r, and for index1 from index0 to the number of columns of matrix_a - 1:
    the index0 element of the index1 element of matrix_r is set to be the dot
    between the index0 element of matrix_q and the index1 element of temp_set,
    then the index1 element of temp_set is set to be the difference of itself and
    the product of the previous element of matrix_r and the index0 element of
    matrix_q. Finally returns [matrix_q, matrix_r].

    Args:
        matrix_a: A matrix of arbitrary length and represented as a list of
                  column vectors.
    Returns:
        matrix_q: The unitary matrix Q with the same length as matrix_a.
                  Represented as a list of column vectors.
        matrix_r: The square upper triangular matrix R with the same length as
                  the rows of matrix_a. Represented as a list of column vectors.
    """
    length = len(matrix_a)
    temp_set = [0] * length
    matrix_q = [[]] * length
    matrix_r = [[]] * length
    for index in range(length):
        matrix_q[index] = [0] * len(matrix_a[0])
    for index in range(length):
        matrix_r[index] = [0] * len(matrix_a)
    for index in range(length):
        temp_set[index] = matrix_a[index]
    for index0 in range(length):
        matrix_r[index0][index0] = twoNorm(temp_set[index0])
        matrix_q[index0] = normalize(temp_set[index0])
        for index1 in range(index0, length):
            matrix_r[index1][index0] = dot(matrix_q[index0], temp_set[index1])
            temp_set[index1] = vectorAdd(temp_set[index1], scalarVectorMulti(
                matrix_q[index0], -1 * matrix_r[index1][index0]))
    return [matrix_q, matrix_r]

def vandermondeFour(vector):
    """ Compute the Vandermonde Matrix of degree 4

    Initilizes result to be a vector with elements 1 and length the same as the input vector. For the second
    element to the last of the input vector the program will define temp to be the vector whose elements are
    the input vector's to the power of the element. Then temp will be appended to the result. Finally
    result is returned.

    Args:
        vector: an arbitrary vector with arbitrary length represented as a list of numbers.

    Returns:
        The vandermonde matrix with dimensions len(vector) x 5 as a list of vectors.
    """
    result = [[1] * len(vector)]
    for index in range(1, 5):
        temp = [x ** index for x in vector]
        result.append(temp)
    return result

def vandermonde(vector, degree):
    """ Compute the Vandermonde Matrix of degree n

    Initilizes result to be the vector of all 1s and length the same as vector. Then appends a column to result
    where each element of this new column is raised to the power of the column number - 1.

    Args:
        vector: an arbitrary vector with arbitrary length represented as a list of numbers.
        degree: a number that signifies the degree for the vandermonde matrix

    Returns:
        The vandermonde matrix stored as a nested list of lists.
    """
    result = [[1] * len(vector)]
    for index in range(1, degree + 1):
        temp = [x ** index for x in vector]
        result.append(temp)
    return result

def backSubSum(matrix, vector, count):
    """ Compute the sum for the Back Substitution Algorithm

    Initilizes result to be 0. Then takes the sum of the product of the elements of vector_x and
    vector_y from count + 1 to length of the vector_x.

    Args:
        vector_x: A single column of the matrix A represented as a list of numbers.
        vector_y: An arbitrary vector with the same length as vector_x represented as a list of numbers.

    Returns:
        A number
    """
    result = 0
    for index in range(len(vector)):
        result += matrix[index][count] * vector[index]
    return result

def backSub(matrix, vector):
    """ Preforms Back Substitution

    Initilizes the result to be the input vector. Then going backwards from the last element of the vector to the
    first, the function sets temp to be the difference between the current element of the vector and the sum of 
    the products of the upper off diagnols of the matrix at the current row and the previously computed elements
    of result.

    Args:
        matrix: An arbitrary upper-triangular matrix represented as a nested list of lists
        vector: An arbitrary vector of compatible dimensions represented as a list of numbers

    Returns:
        A vector represented as a list of numbers, which satisfies the upper-triangular system.
    """
    length = len(vector)
    result = vector
    for index in range(length - 1, -1, -1):
        temp = vector[index] - backSubSum(matrix[index + 1:length], result[index + 1:length], index)
        result[index] = temp / matrix[index][index]
    return result

def alphaPoly(data_x, data_y, degree):
    """ Interpolate data at input_x

    The function will set up a vandermonde matrix as matrix_A, then compute the QR factorization
    of that matrix. Then the function will solve for Q* and use back substitution to solve for and return  the 
    coefficents of the polynomial.

    Args:
        data_x: The independent data stored as a list of numbers
        data_y: The dependent data stored as a list of numbers
        degree: A number that represents the highest degree of the polynomial

    Returns:
        The coefficents of the interpolated polynomial as a list of numbers
    """
    result = 0
    matrix_a = vandermonde(data_x, degree)
    matrix_q, matrix_r = qrFactor(matrix_a)
    inv_matrix_q = conjugateTranspose(matrix_q)
    vector_b = matrixVectorMulti(inv_matrix_q, data_y)
    return backSub(matrix_r, vector_b)

def interpolPoly(data_x, data_y, degree, number_x):
    """ Display the Interpolated Polynomial at number_x

    The function computes the coefficents of the interpolated polynomial and then prints the polynomial and
    the value of the polynomial at the number_x.

    Args:
        vector_alph: a list of numbers that corrispond to the coefficients of the polynomial
        number_x: a number that the polynomial will be evaluated at

    Returns:
        The value of the polynomial at number_x
    """
    vector_alph = alphaPoly(data_x, data_y, degree)
    result = vector_alph[0]
    print("f(x) = ", vector_alph[0], sep="", end="")
    for index in range(1, len(vector_alph)):
        result += vector_alph[index] * number_x ** index
        print(" + ", vector_alph[index], "x^", index, sep="", end="")
    print("\nf(", number_x, ") = ", result, sep="")

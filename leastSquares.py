import basicFunctions as bf

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
    matrix_a = bf.vandermonde(data_x, degree)
    matrix_q, matrix_r = bf.qrFactor(matrix_a)
    inv_matrix_q = bf.conjugateTranspose(matrix_q)
    vector_b = bf.matrixVectorMulti(inv_matrix_q, data_y)
    return bf.backSub(matrix_r, vector_b)

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

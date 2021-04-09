# Calculation of images from two-dimensional points clouds by using a Gaussian
# point spread function
#
import numpy
cimport numpy
cimport cython

from libc.math cimport exp, log, floor

DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t

DTYPE_INT = numpy.int
ctypedef numpy.int_t DTYPE_INT_t

DTYPE_DOUBLE = numpy.double
ctypedef numpy.double_t DTYPE_DOUBLE_t

DTYPE_LONG = numpy.long
ctypedef numpy.long_t DTYPE_LONG_t

ctypedef struct TwoInts:
    int j, i

from libc.math cimport exp, log

cdef:
    double log_2 = 0.69314718055994529
    double exp_A = 1048576 / log_2
    int exp_C = 45799#60801

    union Eco:
        double d
        TwoInts n

    Eco eco

    
cdef double exp_macro(double y):
    eco.n.i = int(exp_A * y + (1072693248 - exp_C))
    return eco.d


cpdef double exp_fast(double x):
    return exp_macro(x)


cpdef double exp_fast2(double x):
    """Slightly slower version, but more accurate."""
    cdef double y = x / 2.0
    return exp_macro(y) / exp_macro(0.0 - y)


# flatten multi-index
cpdef int grid_index(
    int [::1] grid_coords,
    int [::1] shape
):
    cdef int grid_index = grid_coords[0]
    
    grid_index = grid_index * shape[1] + grid_coords[1]

    return grid_index


cpdef int assign_point(
    double [::1] coords,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int [::1] grid_coords
):
    cdef int i
    cdef double y
    
    for i in range(2):
        y = (coords[i]-origin[i]) / spacing
        grid_coords[i] = <int> floor(y)
        
        if (y-grid_coords[i]) > 0.5:
            grid_coords[i] += 1

    return grid_index(grid_coords, shape)


cpdef int add_gaussian(
    double [::1] coords,
    double sigma,
    double weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] values
):

    cdef int i, j, a, b, a2, b2, index, radius
    cdef int [2] grid_coords
    cdef double rho_ij, rho_a, rho_b, delta_rho, w, factor
    cdef double [2] delta
    
    index = assign_point(coords, origin, shape, spacing, grid_coords)

    w = 2 * sigma * sigma
    rho_ij = 0.
    factor = weight / (sigma * sigma)
    
    for i in range(2):
        delta[i] = coords[i] - origin[i] - grid_coords[i] * spacing
        rho_ij += delta[i] * delta[i]
        delta[i] *= 2*spacing/w

    rho_ij *= -1/w
    delta_rho = -spacing * spacing / w

    radius = n_neighbors * n_neighbors
    
    i = grid_coords[0]
    j = grid_coords[1]

    for a in range(-n_neighbors, n_neighbors+1):

        # in grid?        
        if (i+a) < 0 or (i+a) >= shape[0]: continue
        
        # calc contribution
        a2 = a*a
        rho_a = a2 * delta_rho + a * delta[0]

        for b in range(-n_neighbors, n_neighbors+1):

            # in grid?
            if (j+b) < 0 or (j+b) >= shape[1]: continue

            # in neighborhood?
            b2 = b*b
            if a2 + b2 > radius: continue

            # calc contribution
            rho_b = b2 * delta_rho + b * delta[1]

            index = shape[1] * (i+a) + j+b
            values[index] += factor * exp(rho_ij + rho_a + rho_b)
    return 0


cpdef int add_gaussian_fast(
    double [::1] coords,
    double sigma,
    double weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] values
):

    cdef int i, j, a, b, a2, b2, index, radius
    cdef int [2] grid_coords
    cdef double rho_ij, rho_a, rho_b, delta_rho, w, factor
    cdef double [2] delta
    
    index = assign_point(coords, origin, shape, spacing, grid_coords)

    w = 2 * sigma * sigma
    rho_ij = 0.
    factor = weight / (sigma * sigma)
    
    for i in range(2):
        delta[i] = coords[i] - origin[i] - grid_coords[i] * spacing
        rho_ij -= delta[i] * delta[i] / w
        delta[i] *= 2 * spacing / w

    delta_rho = -spacing * spacing / w

    radius = n_neighbors * n_neighbors
    
    i = grid_coords[0]
    j = grid_coords[1]

    for a in range(-n_neighbors, n_neighbors+1):

        # in grid?
        if (i+a) < 0 or (i+a) >= shape[0]: continue
        
        # calc contribution
        a2 = a*a
        rho_a = a2 * delta_rho + a * delta[0]

        for b in range(-n_neighbors, n_neighbors+1):

            # in grid?
            if (j+b) < 0 or (j+b) >= shape[1]: continue

            # in neighborhood?
            b2 = b*b
            if a2 + b2 > radius: continue

            # calc contribution
            rho_b = b2 * delta_rho + b * delta[1]

            index = shape[1] * (i+a) + j+b            
            values[index] += factor * exp_fast(rho_ij+rho_a+rho_b)
    return 0


cpdef int add_gaussian_fast2(
    double [::1] coords,
    double sigma,
    double weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] values
):

    cdef int i, j, a, b, index, radius
    cdef int [2] grid_coords
    cdef double rho_ij, rho_a, rho_b, delta_rho, w
    cdef double [2] delta
    
    index = assign_point(coords, origin, shape, spacing, grid_coords)

    w = 2 * sigma * sigma

    rho_ij = - 3 * log(sigma) + log(weight)

    for i in range(2):
        delta[i] = coords[i] - origin[i] - grid_coords[i] * spacing
        rho_ij -= delta[i] * delta[i] / w
        delta[i] *= 2 * spacing / w

    delta_rho = -spacing * spacing / w

    radius = n_neighbors * n_neighbors

    i = grid_coords[0]
    j = grid_coords[1]

    for a in range(-n_neighbors, n_neighbors+1):

        # in grid?
        if (i+a) < 0 or (i+a) >= shape[0]: continue
        
        # calc contribution
        rho_a = a * (a * delta_rho + delta[0])

        for b in range(-n_neighbors, n_neighbors+1):

            # in grid?
            if (j+b) < 0 or (j+b) >= shape[1]: continue

            # in neighborhood?
            if (a*a + b*b) > radius: continue

            # calc contribution
            rho_b = b * (b * delta_rho + delta[1])

            index = shape[1] * (i+a) + j+b
            values[index] += exp_fast2(rho_ij + rho_a + rho_b)

    return 0


cpdef int update_forces(
    double [::1] coords,
    double sigma,
    double weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] gradient,
    double [::1] forces
):
    cdef int i, j, a, b, index, radius
    cdef int [2] grid_coords
    cdef double rho_ij, rho_a, rho_b, delta_rho, w, value, dx, dy, dz
    cdef double [2] delta
    cdef double [2] dd

    index = assign_point(coords, origin, shape, spacing, grid_coords)

    w = 2 * sigma * sigma
    rho_ij = 0.
    for i in range(2):
        delta[i] = coords[i] - origin[i] - grid_coords[i] * spacing
        rho_ij += delta[i] * delta[i]
        forces[i] = 0.
        dd[i] = 2 * spacing * delta[i] / w

    delta_rho = -spacing * spacing / w

    rho_ij = weight * exp(-rho_ij/w)

    radius = n_neighbors * n_neighbors

    i = grid_coords[0]
    j = grid_coords[1]

    for a in range(-n_neighbors, n_neighbors+1):

        # in grid?
        if (i+a) < 0 or (i+a) >= shape[0]: continue
        
        # calc contribution
        rho_a = a * (a * delta_rho + dd[0])
        dx = delta[0] - spacing*a

        for b in range(-n_neighbors, n_neighbors+1):

            # in grid?
            if (j+b) < 0 or (j+b) >= shape[1]: continue

            # in neighborhood?
            if (a*a + b*b) > radius: continue

            # calc contribution
            rho_b = b * (b * delta_rho + dd[1])
            dy = delta[1] - spacing*b

            index = shape[1] * (i+a) + j+b
            value = exp(rho_a + rho_b) * gradient[index]

            forces[0] -= value * dx
            forces[1] -= value * dy

    for i in range(2):
        forces[i] *= rho_ij

    return 0


cpdef int update_forces_fast(
    double [::1] coords,
    double sigma,
    double weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] gradient,
    double [::1] forces
):
    cdef int i, j, a, b, index, radius
    cdef int [2] grid_coords
    cdef double rho_ij, rho_a, rho_b, delta_rho, w, value, dx, dy
    cdef double [2] delta
    cdef double [2] dd

    index = assign_point(coords, origin, shape, spacing, grid_coords)

    w = 2 * sigma * sigma

    rho_ij = 0.
    for i in range(2):
        delta[i] = coords[i] - origin[i] - grid_coords[i] * spacing
        rho_ij -= delta[i] * delta[i] / w
        forces[i] = 0.
        dd[i] = 2 * spacing / w * delta[i]

    delta_rho = -spacing * spacing / w
    rho_ij = weight * exp_fast(rho_ij)
    radius = n_neighbors * n_neighbors

    i = grid_coords[0]
    j = grid_coords[1]

    for a in range(-n_neighbors, n_neighbors+1):

        # in grid?        
        if (i+a) < 0 or (i+a) >= shape[0]: continue
        
        # calc contribution
        rho_a = a * (a * delta_rho + dd[0])
        dx = delta[0] - spacing * a

        for b in range(-n_neighbors, n_neighbors+1):

            # in grid?
            if (j+b) < 0 or (j+b) >= shape[1]: continue

            # in neighborhood?
            if (a*a + b*b) > radius: continue

            # calc contribution
            rho_b = b * (b * delta_rho + dd[1])
            dy = delta[1] - spacing * b

            index = shape[1] * (i+a) + j+b
            value = exp_fast(rho_a + rho_b) * gradient[index]

            forces[0] -= value * dx
            forces[1] -= value * dy

    for i in range(2):
        forces[i] *= rho_ij

    return 0


cpdef int update_forces_fast2(
    double [::1] coords,
    double sigma,
    double weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] gradient,
    double [::1] forces
):
    cdef int i, j, a, b, index, radius
    cdef int [2] grid_coords
    cdef double rho_ij, rho_a, rho_b, delta_rho, w, value, dx, dy
    cdef double [2] delta
    cdef double [2] dd

    index = assign_point(coords, origin, shape, spacing, grid_coords)

    w = 2 * sigma * sigma
    rho_ij = 0.
    for i in range(2):
        delta[i] = coords[i] - origin[i] - grid_coords[i] * spacing
        rho_ij -= delta[i] * delta[i] / w
        forces[i] = 0.
        dd[i] = 2 * spacing / w * delta[i]

    delta_rho = -spacing * spacing / w
    rho_ij = weight * exp_fast2(rho_ij)
    radius = n_neighbors * n_neighbors

    i = grid_coords[0]
    j = grid_coords[1]

    for a in range(-n_neighbors, n_neighbors+1):

        # in grid?
        if (i+a) < 0 or (i+a) >= shape[0]: continue
        
        # calc contribution
        rho_a = a * (a * delta_rho + dd[0])
        dx = delta[0] - spacing * a

        for b in range(-n_neighbors, n_neighbors+1):

            # in grid?
            if (j+b) < 0 or (j+b) >= shape[1]: continue

            # in neighborhood?
            if (a*a + b*b) > radius: continue

            # calc contribution
            rho_b = b * (b * delta_rho + dd[1])
            dy = delta[1] - spacing * b

            index = shape[1] * (i+a) + j+b
            value = exp_fast2(rho_a + rho_b) * gradient[index]

            forces[0] -= value * dx
            forces[1] -= value * dy

    for i in range(2):
        forces[i] *= rho_ij

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_image(
    double [:,::1] coords, 
    double sigma,
    double [::1] weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] values
):
    cdef Py_ssize_t n
    cdef Py_ssize_t N = len(coords)
    
    for n in range(N):
        add_gaussian(coords[n], sigma, weight[n], origin, shape, spacing,
                     n_neighbors, values)

        
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_image_fast(
    double [:,::1] coords, 
    double sigma,
    double [::1] weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] values
):
    cdef Py_ssize_t n    
    for n in range(len(coords)):
        add_gaussian_fast(coords[n], sigma, weight[n], origin, shape, spacing,
                          n_neighbors, values)

        
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_image_fast2(
    double [:,::1] coords, 
    double sigma,
    double [::1] weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] values
):
    cdef Py_ssize_t n
    for n in range(len(coords)):
        add_gaussian_fast2(coords[n], sigma, weight[n], origin, shape, spacing,
                           n_neighbors, values)

        
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_forces(
    double [:, ::1] coords, 
    double sigma,
    double [::1] weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] gradient,
    double [:, ::1] forces
):
    cdef Py_ssize_t n
    cdef double factor = sigma
    
    grad = numpy.zeros(2, dtype=DTYPE_DOUBLE)
    
    factor *= factor
    factor *= factor
    factor = 1./factor

    for n in range(len(coords)):

        grad[:] = 0.
        
        update_forces(
            coords[n],
            sigma,
            weight[n],
            origin,
            shape,
            spacing,
            n_neighbors,
            gradient,
            grad
        )
        forces[n, 0] += factor * grad[0]
        forces[n, 1] += factor * grad[1]

        
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_forces_fast(
    double [:,::1] coords, 
    double sigma,
    double [::1] weight,
    double [::1] origin,
    int [::1] shape,
    double spacing,
    int n_neighbors, 
    double [::1] gradient,
    double [:,::1] forces
):
    cdef Py_ssize_t n
    cdef Py_ssize_t N = len(coords)

    cdef double factor = sigma

    factor *= factor
    factor *= factor * sigma
    factor  = 1. / factor

    for n in range(N):

        update_forces_fast(coords[n],
                           sigma,
                           weight[n],
                           origin,
                           shape,
                           spacing,
                           n_neighbors,
                           gradient,
                           forces[n])

        forces[n,0] *= factor
        forces[n,1] *= factor

        
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_forces_fast2(double [:,::1] coords, 
                      double sigma,
                      double [::1] weight,
                      double [::1] origin,
                      int [::1] shape,
                      double spacing,
                      int n_neighbors, 
                      double [::1] gradient,
                      double [:,::1] forces):

    cdef Py_ssize_t n
    cdef Py_ssize_t N = len(coords)

    cdef double factor = sigma

    factor *= factor
    factor *= factor * sigma
    factor  = 1. / factor

    for n in range(N):

        update_forces_fast2(coords[n],
                            sigma,
                            weight[n],
                            origin,
                            shape,
                            spacing,
                            n_neighbors,
                            gradient,
                            forces[n])

        forces[n,0] *= factor
        forces[n,1] *= factor


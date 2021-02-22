import numpy
import torch

__all__ = ['find_coefficients', 'transform']


def cdist(K: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance between K[i, :] and B[j, :].

    Arguments
    ---------
        K : torch.Tensor
        B : torch.Tensor
    """
    assert K.ndim == 3
    assert B.ndim == 3


    K = K.unsqueeze(2)
    B = B.unsqueeze(1)
    D = K - B
    return torch.norm(D, dim=3)


def pairwise_radial_basis(K: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute the TPS radial basis function phi(r) between every row-pair of K
    and B where r is the Euclidean distance.

    Arguments
    ---------
        K : torch.Tensor
            n by d vector containing n d-dimensional points.
        B : torch.Tensor
            m by d vector containing m d-dimensional points.

    Return
    ------
        P : torch.Tensor
            n by m matrix where.
            P(i, j) = phi( norm( K(i,:) - B(j,:) ) ),
            where phi(r) = r^2*log(r), if r >= 1
                           r*log(r^r), if r <  1
    """
    # r_mat(i, j) is the Euclidean distance between K(i, :) and B(j, :).
    r_mat = cdist(K, B)

    pwise_cond_ind1 = r_mat >= 1
    pwise_cond_ind2 = r_mat < 1
    r_mat_p1 = r_mat[pwise_cond_ind1]
    r_mat_p2 = r_mat[pwise_cond_ind2]

    # P correcponds to the matrix K from [1].
    P = torch.zeros(r_mat.shape, dtype=torch.float64, device=K.device)
    P[pwise_cond_ind1] = (r_mat_p1**2) * torch.log(r_mat_p1)
    P[pwise_cond_ind2] = r_mat_p2 * torch.log(torch.pow(r_mat_p2, r_mat_p2))

    return P


def find_coefficients(control_points: torch.Tensor,
                      target_points: torch.Tensor,
                      lambda_: float = 0.,
                      solver: str = 'exact') -> torch.Tensor:
    """Given a set of control points and their corresponding points, compute the
    coefficients of the TPS interpolant deforming surface.

    Arguments
    ---------
        control_points : torch.Tensor
            p by d vector of control points
        target_points : torch.Tensor
            p by d vector of corresponding target points on the deformed
            surface
        lambda_ : float
            regularization parameter
        solver : str
            the solver to get the coefficients. default is 'exact' for the exact
            solution. Or use 'lstsq' for the least square solution.

    Return
    ------
        coef : torch.Tensor
            the coefficients

    .. seealso::

        http://cseweb.ucsd.edu/~sjb/pami_tps.pdf
    """
    # ensure data type and shape
    if control_points.shape != target_points.shape:
        raise ValueError(
            'Shape of and control points {cp} and target points {tp} are not the same.'.
            format(cp=control_points.shape, tp=target_points.shape))

    bs, p, d = control_points.shape

    # The matrix
    K = pairwise_radial_basis(control_points, control_points)
    P = torch.cat([torch.ones((bs, p, 1), device=K.device, requires_grad=True), control_points], dim=2)

    # Relax the exact interpolation requirement by means of regularization.
    K = K + lambda_ * torch.eye(p, device=K.device)

    # Target points
    M = torch.cat([
        torch.cat([K, P], dim=2),
        torch.cat([P.permute(0,2,1), torch.zeros((bs, d + 1, d + 1), device=K.device, requires_grad=True)], dim=2)
    ], dim=1)
    Y = torch.cat([target_points, torch.zeros((bs, d + 1, d), device=K.device, requires_grad=True)], dim=1)

    # solve for M*X = Y.
    # At least d+1 control points should not be in a subspace; e.g. for d=2, at
    # least 3 points are not on a straight line. Otherwise M will be singular.
    solver = solver.lower()
    if solver == 'exact':
        X, _ = torch.solve(Y,M)

    else:
        raise ValueError('Unknown solver: ' + solver)

    return X


def transform(source_points: torch.Tensor, control_points: torch.Tensor,
              coefficient: torch.Tensor) -> torch.Tensor:
    """Transform the source points form the original surface to the destination
    (deformed) surface.

    Arguments
    ---------
        source_points : torch.Tensor
            n by d array of source points to be transformed
        control_points : torch.Tensor
            the control points used in the function `find_coefficients`
        coefficient : torch.Tensor
            the computed coefficients

    Return
    ------
        deformed_points : torch.Tensor
            n by d array of the transformed point on the target surface
    """
    if source_points.shape[-1] != control_points.shape[-1]:
        raise ValueError(
            'Dimension of source points ({sd}D) and control points ({cd}D) are not the same.'.
            format(sd=source_points.shape[-1], cd=control_points.shape[-1]))

    bs, n = source_points.shape[:2]

    A = pairwise_radial_basis(source_points, control_points)
    K = torch.cat([A, torch.ones((bs, n, 1),
        device=A.device, requires_grad=True), source_points], dim=2)

    deformed_points = K@coefficient
    return deformed_points

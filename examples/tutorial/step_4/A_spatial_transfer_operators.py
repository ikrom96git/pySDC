import numpy as np
from collections import namedtuple

from implementations.problem_classes.HeatEquation_1D_FD import heat1d
from implementations.datatype_classes.mesh import mesh
from implementations.transfer_classes.TransferMesh_1D import mesh_to_mesh_1d_dirichlet
from examples.tutorial.step_1.B1_spatial_accuracy_check import get_accuracy_order

# setup id for gathering the results (will sort by nvars)
ID = namedtuple('ID', 'nvars_fine')


def main():
    """
    A simple test program to test interpolation order in space
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 3  # frequency for the test value

    # initialize transfer parameters
    space_transfer_params = {}
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 4

    nvars_fine_list = [2**p-1 for p in range(5,10)]

    # set up dictionary to store results (plus lists)
    results = {}
    results['nvars_list'] = nvars_fine_list

    for nvars_fine in nvars_fine_list:

        print('Working on nvars_fine = %4i...' %(nvars_fine))

        # instantiate fine problem
        problem_params['nvars'] = nvars_fine  # number of degrees of freedom
        Pfine = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

        # instantiate coarse problem using half of the DOFs
        problem_params['nvars'] = int((nvars_fine + 1) / 2.0 - 1)
        Pcoarse = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

        # instantiate spatial interpolation
        T = mesh_to_mesh_1d_dirichlet(fine_prob=Pfine, coarse_prob=Pcoarse, params=space_transfer_params)

        # set exact fine solution to compare with
        xvalues_fine = np.array([(i + 1) * Pfine.dx for i in range(Pfine.params.nvars)])
        uexact_fine = Pfine.dtype_u(0)
        uexact_fine.values = np.sin(np.pi * Pfine.params.freq * xvalues_fine)

        # set exact coarse solution as source
        xvalues_coarse = np.array([(i + 1) * Pcoarse.dx for i in range(Pcoarse.params.nvars)])
        uexact_coarse = Pfine.dtype_u(0)
        uexact_coarse.values = np.sin(np.pi * Pcoarse.params.freq * xvalues_coarse)

        # do the interpolation/prolongation
        uinter = T.prolong(uexact_coarse)

        # compute error and store
        id = ID(nvars_fine=nvars_fine)
        results[id] = abs(uinter-uexact_fine)

    print('Running order checks...')
    orders = get_accuracy_order(results)
    for p in range(len(orders)):
        print('Expected order %2i, got order %5.2f, deviation of %5.2f%%'\
              %(space_transfer_params['iorder'], orders[p], 100*abs(space_transfer_params['iorder']-orders[p])/space_transfer_params['iorder']))
        assert abs(space_transfer_params['iorder']-orders[p])/space_transfer_params['iorder'] < 0.05, \
            'ERROR: did not get expected orders for interpolation, got %s' %str(orders[p])
    print('...got what we expected!')


if __name__ == "__main__":
    main()
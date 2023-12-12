import tpot2
import sklearn.metrics
import sklearn
import argparse
import utils_base
import sys

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # number of threads to use during estimator evalutation
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    # where to save the results/models
    parser.add_argument("-s", "--savepath", default="binary_results", required=False, nargs='?')
    # number of total replicates for each experiment
    parser.add_argument("-r", "--num_reps", default=1, required=False, nargs='?')
    # number of total replicates for each experiment
    parser.add_argument("-ss", "--sel_scheme", default=0, required=False, nargs='?')
    schemes = ['lexicase','tournament','random']

    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    save_dir = args.savepath
    num_reps = int(args.num_reps)

    if int(args.sel_scheme) < 0 or len(schemes) <= int(args.sel_scheme):
        sys.exit('MAIN: INVALID SCHEME TO RUN')
    scheme = schemes[int(args.sel_scheme)]


    # task_id_lists = [359990, 360112, 189354, 7593, 189843, 273, 359960, 189836, 75127, 168796, 167181, 75193, 168794, 189871, 189873, 189874, 189908, 189909,]
    task_id_lists = [167104, 167184, 167168, 167161, 167185, 189905, 167152, 167181, 189906, 189862, 167149, 189865, 167190, 189861, 189872, 168794, 189871, 168796, 168797, 75097, 126026, 189909, 126029, 126025, 75105, 168793, 189874, 167201, 189908, 189860, 168792, 167083, 167200, 168798, 189873, 189866, 75127, 75193]

    utils_base.loop_through_tasks(scheme, task_id_lists, save_dir, num_reps, n_jobs)


if __name__ == '__main__':
    main()
    print('END OF MAIN')

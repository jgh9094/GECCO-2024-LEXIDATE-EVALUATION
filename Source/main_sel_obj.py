import argparse
import utils_sel_obj

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # number of threads to use during estimator evalutation
    parser.add_argument("-n", "--n_jobs", default=30,  required=False, nargs='?')
    # where to save the results/models
    parser.add_argument("-s", "--savepath", default="./", required=False, nargs='?')
    # number of total replicates for each experiment
    parser.add_argument("-r", "--num_reps", default=1, required=False, nargs='?')
    # proportion of training data to split between learning and selecting
    parser.add_argument("-pr", "--proportion", default=0.0, required=False, nargs='?')
    # seed to start at
    parser.add_argument("-so", "--seed_offset", default=0.0, required=False, nargs='?')

    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    save_dir = args.savepath
    num_reps = int(args.num_reps)
    seed_offset = int(args.seed_offset)
    proportion = float(args.proportion)
    scheme = 'lexicase'

    task_id_lists = [167104, 167184, 167168, 167161, 167185, 189905]

    utils_sel_obj.loop_through_tasks(scheme, task_id_lists, save_dir, num_reps, n_jobs, proportion, seed_offset)


if __name__ == '__main__':
    main()
    print('END OF MAIN')
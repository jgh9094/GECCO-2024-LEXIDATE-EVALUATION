import os
import argparse
import sys
import pickle as pkl

GENERATIONS = 200

def ExperimentDir(exp):
    if exp == 0:
        return 'Base/'
    elif exp == 1:
        return 'Lex-50/'
    elif exp == 2:
        return 'Lex-30/'
    elif exp == 3:
        return 'Lex-10/'
    else:
        sys.exit('UTILS: INVALID EXPERIMENT DIR TO FIND')

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # where to save the results/models
    parser.add_argument("-d", "--data_dir", default="./", required=False, nargs='?')
    # number of total replicates for each experiment
    parser.add_argument("-r", "--num_reps", default=40, required=False, nargs='?')
    # seed we are starting from for each experiment
    parser.add_argument("-s", "--seed", default=0, required=False, nargs='?')
    # experiment we want to get data for
    parser.add_argument("-e", "--experiment", default=0, required=False, nargs='?')

    args = parser.parse_args()
    data_dir = args.data_dir
    num_reps = int(args.num_reps)
    seed = int(args.seed)
    exp_dir = ExperimentDir(exp=int(args.experiment))

    print('EXPERIMENT DIR:', data_dir + exp_dir)
    task_id_lists = [167104, 167184, 167168, 167161, 167185, 189905]

    FAILED_FILES = []
    EMPTY_DIRECTORIES = []
    UNFINISHED_RUNS = []
    for task_pos, task in enumerate(task_id_lists):
        task_limit = False
        for rep in range(num_reps):
            dir = data_dir + exp_dir + str(rep + seed + (task_pos * num_reps)) + '-' + str(task) + '/'

            # last folder we made it to
            if os.path.isdir(dir) is False:
                print('REPS:',dir)
                task_limit = True
                break

            # check if data file exists
            data_pkl = dir + 'data.pkl'
            failed_pkl = dir + 'failed.pkl'
            evaluated_pkl = dir + 'evaluated_individuals.pkl'
            scores_pkl = dir + 'scores.pkl'
            fitted_pkl = dir + 'fitted_pipeline.pkl'

            # folder is empty
            if not any(os.scandir(dir)):
                print(dir,': EMPTY')
                continue

            # failed runs
            if os.path.exists(failed_pkl):
                print(dir,': FAILED.PKL')
                FAILED_FILES.append(dir)
                continue

            # check if data csv reached generation expectation and not empty
            df = pkl.load(open(data_pkl,'rb'))
            if df.empty:
                print(dir,': DATA.PKL EMPTY')
                UNFINISHED_RUNS.append(dir)
                continue

            if max(df['gen'].to_list()) != GENERATIONS:
                print(f"{dir}: {max(df['gen'].to_list())} GEN REACHED")
                UNFINISHED_RUNS.append(dir)
                continue

        if task_limit:
            print('FINAL TASK REACHED:', task_id_lists[task_pos - 1])
            print('TOTAL NUMBER OF TASKS DONE:', task_pos + 1)
            break

    print()
    print('-'*150)
    print()

    print('FAILED FILES:')
    for err in FAILED_FILES:
        print(err)
    print('\nUNFINISHED RUNS:')
    print()
    for err in UNFINISHED_RUNS:
        print(err)
    print('\EMPTY DIRS:')
    print()
    for err in EMPTY_DIRECTORIES:
        print(err)


if __name__ == '__main__':
    main()
    print('FINISHED CHECKING RUNS')
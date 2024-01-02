import os
import argparse
import sys
import pickle as pkl

GENERATIONS = 200

def ExperimentDir(exp):
    if exp == 0:
        return 'Base/'
    elif exp == 1:
        return '10/Lexicase/'
    elif exp == 2:
        return '30/Lexicase/'
    elif exp == 3:
        return '50/Lexicase/'
    else:
        sys.exit('UTILS: INVALID EXPERIMENT DIR TO FIND')

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # where to save the results/models
    parser.add_argument("-d", "--data_dir", default="./", required=False, nargs='?')
    # number of total replicates for each experiment
    parser.add_argument("-r", "--num_reps", default=30, required=False, nargs='?')
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
    task_id_lists = [167104, 167184, 167168, 167161, 167185, 189905, 167152, 167181, 189906, 189862, 167149, 189865, 167190, 189861, 189872,
                     168794, 189871, 168796, 168797, 75097, 126026, 189909, 126029, 126025, 75105, 168793, 189874, 167201, 189908, 189860, 168792,
                     167083, 167200, 168798, 189873, 189866, 75127, 75193]

    FAILED_FILES = []
    UNFINISHED_RUNS = []
    for task_pos, task in enumerate(task_id_lists):
        task_limit = False
        for rep in range(num_reps):
            dir = data_dir + exp_dir + str(task) + '-' + str(rep + seed + (task_pos * num_reps)) + '/'

            print(dir)

            # whats the lastest folder we made it to
            if os.path.isdir(dir) is False:
                print('REPS:',dir)
                task_limit = True
                break

            # check if data file exists
            data_pkl = dir + 'data.pkl'
            failed_pkl = dir + 'failed.pkl'

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
                # print(dir,': 50K GENS NOT REACHED')
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





if __name__ == '__main__':
    main()
    print('FINISHED CHECKING RUNS')

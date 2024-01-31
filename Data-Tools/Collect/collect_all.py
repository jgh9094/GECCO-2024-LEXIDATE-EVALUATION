import tpot2
import sklearn.metrics
import sklearn
import sys
import pickle as pkl
import pandas as pd
import os

# display stuff
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# variables used throughout
data_dir = '../../Results/'
replicates = 30
exp_dirs = ['Base/', 'Lex-10/','Lex-30/','Lex-50/']
seeds = [5000,5300,5600,5900]
proportions = [0.0, 0.1, 0.3, 0.5]
data_pkl = '/data.pkl'
scores_pkl = '/scores.pkl'
failed_pkl = '/failed.pkl'
evaluated_pkl = '/evaluated_individuals.pkl'
fitted_pkl = '/fitted_pipeline.pkl'
scores_key = ['train_auroc','train_accuracy','train_balanced_accuracy','train_logloss','auroc','accuracy','balanced_accuracy','logloss','taskid','selection','seed']


def Complexity():
    print('COMPLEXITY')
    # data holders
    COMPLEXITY = []
    ACRO = []
    SEED = []
    TASK = []

    for cur_dir,seed, proportion in zip(exp_dirs,seeds, proportions):

        dir = data_dir + cur_dir

        if proportion == 0.0:
            acro = '10-f cv'
        elif proportion == 0.1:
            acro = '90/10'
        elif proportion == 0.3:
            acro = '70/30'
        elif proportion == 0.5:
            acro = '50/50'

        print('dir:',dir)
        print('acro:',acro)

        for subdir, dirs, files in os.walk(dir):
            # skip root dir
            if subdir == dir:
                continue
            # skip failed (will fix on the go)
            if os.path.exists(subdir + failed_pkl):
                continue

            # folder is empty
            if os.path.exists(subdir + failed_pkl) is False and os.path.exists(subdir + data_pkl) is False and \
                os.path.exists(subdir + evaluated_pkl) is False and os.path.exists(subdir + scores_pkl) is False and \
                os.path.exists(subdir + fitted_pkl) is False:
                continue

            # scores dictionary data (swap wit upper block if we care about seed, then pass seed down)
            fiitted_pipeline = pkl.load(open(subdir + fitted_pkl,'rb'))

            SEED.append(subdir.split('/')[-1].split('-')[0])
            TASK.append(subdir.split('/')[-1].split('-')[1])
            ACRO.append(acro)
            COMPLEXITY.append(float(tpot2.objectives.complexity_scorer(fiitted_pipeline,0,0)))

    pd.DataFrame({'seed': pd.Series(SEED),'taskid': pd.Series(TASK),'acro': pd.Series(ACRO),'complexity':pd.Series(COMPLEXITY)}).to_csv(path_or_buf='./complexity.csv', index=False)

def Scores():
    print('SCORES')
    # data holders
    SCORES_DF = []
    OVER_TIME_DF = []

    for cur_dir,seed, proportion in zip(exp_dirs,seeds, proportions):

        dir = data_dir + cur_dir

        if proportion == 0.0:
            acro = '10-f cv'
        elif proportion == 0.1:
            acro = '90/10'
        elif proportion == 0.3:
            acro = '70/30'
        elif proportion == 0.5:
            acro = '50/50'

        print('dir:',dir)
        print('acro:',acro)

        for subdir, dirs, files in os.walk(dir):
            # skip root dir
            if subdir == dir:
                continue
            # skip failed (will fix on the go)
            if os.path.exists(subdir + failed_pkl):
                continue

            # folder is empty
            if os.path.exists(subdir + failed_pkl) is False and os.path.exists(subdir + data_pkl) is False and \
                os.path.exists(subdir + evaluated_pkl) is False and os.path.exists(subdir + scores_pkl) is False and \
                os.path.exists(subdir + fitted_pkl) is False:
                continue

            # scores dictionary data (swap wit upper block if we care about seed, then pass seed down)
            scores_dict = pkl.load(open(subdir + scores_pkl,'rb'))
            scores_df = pd.DataFrame([scores_dict],  columns=scores_dict.keys())[scores_key]
            scores_df['proportion'] = proportion
            scores_df['acro'] = acro
            SCORES_DF.append(scores_df)

            # over time data
            data_df = pkl.load(open(subdir + data_pkl,'rb'))
            data_df['taskid'] = scores_dict['taskid']
            data_df['proportion'] = proportion
            data_df['acro'] = acro
            data_df['seed'] = subdir.split('/')[-1].split('-')[0]
            OVER_TIME_DF.append(data_df)


    pd.concat(OVER_TIME_DF).to_csv(path_or_buf='ot_data.csv', index=False)
    pd.concat(SCORES_DF).to_csv(path_or_buf='scores.csv', index=False)

def main():
    Scores()
    Complexity()


if __name__ == '__main__':
    main()
    print('FINISHED COLLECTING')
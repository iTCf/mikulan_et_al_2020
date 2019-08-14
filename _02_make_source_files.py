import sys
from info import dir_base, subjects_dir
from fx_source_loc import make_bem_and_source_space, make_forward_solution


def make_source_files(subject):
    print('Making source files subject: %s - Started' % subject)

    # make source files
    make_bem_and_source_space(subject, subjects_dir, dir_base)

    # todo: add stop to wait for -trans.fif creation
    make_forward_solution(subject, subjects_dir, dir_base)


if __name__ == '__main__':
    subj = sys.argv[1]
    make_source_files(subj)

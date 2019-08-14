from info import subjects_dir, dir_base
from fx_source_loc import make_anony_fwd
import sys

if __name__ == '__main__':
    subj = sys.argv[1]
    print('Making anonymized forward models - subject: %s' % subj)
    make_anony_fwd(subj, dir_base, subjects_dir)


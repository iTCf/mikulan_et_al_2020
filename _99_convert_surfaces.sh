#!/usr/bin/env bash

dir_in=$1 # freesurfer's subjects dir
dir_out=$2 # base directory for saving

for i in {1..7}
do
    subj=sub-0${i}
    for b in brain inner_skull outer_skull outer_skin
    do
        file_in=${dir_in}/s${i}/bem/watershed/s${i}_${b}_surface
        file_out=${dir_out}/derivatives/sourcemodelling/${subj}/anat/${subj}_${b}.surf.gii
        mris_convert ${file_in} ${file_out}
    done

    for h in rh lh
    do
        echo ${h}
        if [[ "${h}" == "rh" ]]
        then
            hemi=R
        else
            hemi=L
        fi

        for surf in pial inflated
        do
            file_in=${dir_in}/s${i}/surf/${h}.pial
            file_out=${dir_out}/derivatives/sourcemodelling/${subj}/anat/${subj}_hemi-${hemi}_${surf}.surf.gii
#            echo $file_in
#            echo $file_out
            mris_convert ${file_in} ${file_out}
        done
    done
done

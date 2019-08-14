#!/usr/bin/env bash

while getopts ":i:o:n:" opt; do
  case $opt in
    i) in_dir="$OPTARG"
    ;;
    o) out_dir="$OPTARG"
    ;;
    n) name="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

echo ${in_dir}
echo ${out_dir}
echo ${name}


## defaced
defaced=${in_dir}/T1_pydefaced.nii
mri_watershed -useSRAS -surf ${out_dir}/defaced/${name}_defaced ${defaced} ${out_dir}/defaced/${name}_defaced_ws

# mask face
mf=${in_dir}/T1_mf.nii
mri_watershed -useSRAS -surf ${out_dir}/mf/${name}_mf ${mf} ${out_dir}/mf/${name}_mf_ws


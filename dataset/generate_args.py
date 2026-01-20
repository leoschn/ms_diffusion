import glob
import os

sample_list = glob.glob('/lustre/fsn1/projects/rech/bun/ucg81ws/mzml/**.mzML', recursive=True)
i=0
for sample in sample_list:
    f_name = os.path.basename(sample).split('.mzML')[0]
    with open('/lustre/fswork/projects/rech/bun/ucg81ws/these/ms_diffusion/img_creation_dir/exec_{i}.txt'.format(i=i), 'w') as f:
        f.write(f_name)
        i+=1
print(i)

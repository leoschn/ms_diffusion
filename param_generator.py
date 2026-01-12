import os.path as path
from pathlib import Path

root = Path("./data/raw/entero")

files = list(root.rglob("*100vW_100SPD.mzML"))
files_2 = list(root.rglob("*-d200.mzML"))
species = []
print(len(files))
for f_name in files:
    split_path = str(f_name).split("/")
    specie = split_path[-1].split("-")[0]
    if specie!= 'Pool purs_100vW_100SPD.mzML':
        species.append(specie)

for f_name in files_2:
    split_path = str(f_name).split("/")
    specie = split_path[-1].split("-")[0]
    if specie!= 'Pool purs_100vW_100SPD.mzML':
        species.append(specie)
print(list(set(species)))
i=0
for spe in list(set(species)):
    with open('cfg_dir/exec_{i}.txt'.format(i=i), 'w') as f:
        f.write(
            "--lib '' --threads 79 --verbose 1 --out /work/output/report_lib_{specie}.tsv --qvalue 0.01 --matrices --out-lib /work/output/lib_{specie}.parquet --gen-spec-lib --predictor --fasta /work/data/fasta/{specie}.fasta  --fasta /work/data/fasta/240711_prot_sang_human.fasta  --fasta /work/data/fasta/0602_Universal Contaminants.fasta --fasta-search --min-fr-mz 200 --max-fr-mz 1800 --min-pep-len 7 --max-pep-len 30 --min-pr-mz 350 --max-pr-mz 1250 --min-pr-charge 2 --max-pr-charge 4 --cut K*,R* --missed-cleavages 1 --var-mods 1 --var-mod UniMod:35,15.994915,M --reanalyse --relaxed-prot-inf --rt-profiling".format(specie=spe)
)
        print(spe)
    i+=1

print(i,' different specie lib to generate')
for f_name in files:
    split_path = str(f_name).split("/")
    name = split_path[-1].split(".")[0]
    specie = split_path[-1].split("-")[0]
    if specie!= 'Pool purs_100vW_100SPD.mzML':
        rel_path = path.join('/work/data/raw/',split_path[-3],split_path[-2], split_path[-1])
        with open('cfg_dir/exec_{i}.txt'.format(i=i), 'w') as f:
            f.write(
                "--f {rel_path} --lib /work/output/lib_{specie}.parquet --threads 79 --verbose 1 --out /work/output/report_{f_name}.tsv --qvalue 0.01 --matrices --out-lib /work/output/lib_{f_name}.parquet --gen-spec-lib --predictor  --var-mods 1 --var-mod UniMod:35,15.994915,M --reanalyse --relaxed-prot-inf --rt-profiling".format(specie=specie, rel_path=rel_path,f_name=name)

            )
        i+=1

for f_name in files_2:
    split_path = str(f_name).split("/")
    name = split_path[-1].split(".")[0]
    specie = split_path[-1].split("-")[0]
    if specie!= 'Pool purs_100vW_100SPD.mzML':
        rel_path = path.join('/work/data/raw/',split_path[-3],split_path[-2], split_path[-1])
        with open('cfg_dir/exec_{i}.txt'.format(i=i), 'w') as f:
            f.write(
                "--f {rel_path} --lib /work/output/lib_{specie}.parquet --threads 79 --verbose 1 --out /work/output/report_{f_name}.tsv --qvalue 0.01 --matrices --out-lib /work/output/lib_{f_name}.parquet --gen-spec-lib --predictor  --var-mods 1 --var-mod UniMod:35,15.994915,M --reanalyse --relaxed-prot-inf --rt-profiling".format(specie=specie, rel_path=rel_path,f_name=name)

            )
        i+=1

print(i,' total number of jobs to run')


template=$1
python -m lconvnet.template.generate_runs --dir $template
python -m lconvnet.slurm.generate_sbatch_script --dir $template
python -m lconvnet.slurm.generate_sbatch_script --dir $template --resume --test --out batch_run_resume_test.sh
python -m lconvnet.slurm.generate_sbatch_script --dir $template --resume --out batch_run_resume.sh
python -m lconvnet.slurm.generate_sbatch_script --dir $template --test --out batch_run_test.sh

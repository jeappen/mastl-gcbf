# Run a plangcbf+ on different number of agents and obstacles in a given environment
echo "Usage: ./run_plangcbf+.sh <model_path>"
echo "Model path: $1"

num_agents=(8 16 32)
# num_obs=(0 4)
num_obs=(0)  # Ignore Obstacles for now
n_epi=5
model_path="$1" #"./logs/SingleIntegrator/plangcbf+/seed0_20240418152958"
for num_agent in ${num_agents[@]}; do
  for num_ob in ${num_obs[@]}; do
    echo "Running $num_agent agents, $num_ob obstacles, $spec"
    python test.py --path $model_path --epi $n_epi --area-size 4 -n $num_agent --obs $num_ob --nojit-rollout --log  --async-planner --plot-async-change  --ignore-on-finish
  done
done

bash scripts/test_log_2_summary.sh ${model_path}

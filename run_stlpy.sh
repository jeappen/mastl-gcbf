# Run stlpy on SingleIntegrator environment
num_agents=(8 16 32)
num_obs=(0 4)
specs=("seq2" "seq3" "cover2" "cover3")
spec_len=30
n_epi=10
model_path="$1" #"./pretrained/SingleIntegrator/gcbf+/"
for num_agent in ${num_agents[@]}; do
  for num_ob in ${num_obs[@]}; do
    for spec in ${specs[@]}; do
      echo "Running $num_agent agents, $num_ob obstacles, $spec"
      python test.py --path $model_path --epi $n_epi --area-size 4 -n $num_agent --obs $num_ob --nojit-rollout --planner stlpy --spec-len $spec_len --goal-sample-interval 10 --spec $spec --log
    done
  done
done

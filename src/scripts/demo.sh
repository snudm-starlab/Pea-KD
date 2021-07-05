python ../finetune.py \
--task 'MRPC' \
--train_type 'ft' \
--model_type 'Original' \
--student_hidden_layers 12 \
--saving_criterion_acc 1.0 \
--saving_criterion_loss 1.0 \
--output_dir 'run-1'\

python ../save_teacher_outputs.py \

python ../PTP.py \
--task 'MRPC' \
--train_type 'ft' \
--model_type 'SPS' \
--student_hidden_layer 3 \
--saving_criterion_loss 100 \
--output_dir 'run-1'\

python ../finetune.py \
--task 'MRPC' \
--train_type 'pkd' \
--model_type 'SPS' \
--student_hidden_layers 3 \
--saving_criterion_acc 1.0 \
--saving_criterion_loss 0.6 \
--load_model_dir 'run-1/PTP.encoder_loss.pkl' \
--output_dir 'run-1/final_results'\
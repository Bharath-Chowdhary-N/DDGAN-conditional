import os
command = "python3 test_ddgan.py --dataset custom --image_size 64 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2 --epoch_id 1050 --z_emb_dim 64 --batch_size 184"
for ite in range(50):
    os.system(command)
    print("This is new file")
    print("{} iteration is completed".format(ite))

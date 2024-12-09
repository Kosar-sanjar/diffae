# run_training.py

from templates import *
from choices import TrainMode  # Import TrainMode from choices.py
# from templates_latent import *  # Deprecated; remove or comment out
import torch

def main():
    # ---------------------------
    # Step 1: Train the Autoencoder (Semantic Encoder)
    # ---------------------------
    print("Step 1: Training the Autoencoder (Semantic Encoder)")
    gpus_encoder = [0, 1, 2, 3]  # Adjust based on available GPUs
    conf_encoder = ffhq128_autoenc_130M()
    # train_mode is already set within ffhq128_autoenc_130M()
    train(conf_encoder, gpus=gpus_encoder)
    
    # ---------------------------
    # Step 2: Infer Latents for Latent DPM Training
    # ---------------------------
    print("Step 2: Inferring Latents for Latent DPM Training")
    gpus_infer = [0, 1, 2, 3]  # Can utilize multiple GPUs for faster inference
    conf_infer = ffhq128_autoenc_130M()  # Use the same configuration as encoder
    conf_infer.eval_programs = ['infer']  # Specify the inference program
    train(conf_infer, gpus=gpus_infer, mode='eval')
    
    # ---------------------------
    # Step 3: Train the Latent DPM
    # ---------------------------
    print("Step 3: Training the Latent DPM")
    gpus_latent = [0]  # Typically requires fewer resources
    conf_latent = ffhq128_autoenc_latent()  # Define a separate configuration for latent training
    train(conf_latent, gpus=gpus_latent)
    
    # ---------------------------
    # Step 4: Unconditional Sampling and Evaluation
    # ---------------------------
    print("Step 4: Unconditional Sampling and Evaluation")
    gpus_eval = [0, 1, 2, 3]  # Utilize multiple GPUs for efficiency
    conf_eval = ffhq128_autoenc_latent()  # Use the same latent configuration
    conf_eval.eval_programs = ['fid(10,10)']  # Specify the evaluation program
    train(conf_eval, gpus=gpus_eval, mode='eval')

if __name__ == '__main__':
    main()

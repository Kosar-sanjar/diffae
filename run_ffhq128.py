from experiment import train  # Import the train function from experiment.py
from templates import *       # Import all predefined configurations
from choices import TrainMode  # Import TrainMode from choices.py
# from templates_latent import *  # Deprecated; remove or comment out
import torch

def main():
    # ---------------------------
    # Step 1: Train the Autoencoder (Semantic Encoder)
    # ---------------------------
    print("Step 1: Training the Autoencoder (Semantic Encoder)")
    gpus_encoder = [0, 1, 2, 3]  # Adjust based on available GPUs
    conf_encoder = ffhq128_autoenc_130M()  # Load the autoencoder configuration
    # train_mode is already set within ffhq128_autoenc_130M()
    train(conf_encoder, gpus=gpus_encoder, nodes=1, mode='train')
    
    # ---------------------------
    # Step 2: Infer Latents for Latent DPM Training
    # ---------------------------
    print("\nStep 2: Inferring Latents for Latent DPM Training")
    gpus_infer = [0, 1, 2, 3]  # Utilize multiple GPUs for faster inference
    conf_infer = ffhq128_autoenc_130M()  # Use the same configuration as the encoder
    conf_infer.eval_programs = ['infer']  # Specify the inference program
    train(conf_infer, gpus=gpus_infer, nodes=1, mode='eval')
    
    # ---------------------------
    # Step 3: Train the Latent DPM
    # ---------------------------
    print("\nStep 3: Training the Latent DPM")
    gpus_latent = [0]  # Typically requires fewer resources
    conf_latent = ffhq128_autoenc_latent()  # Load the latent DPM configuration
    # train_mode is already set within ffhq128_autoenc_latent()
    train(conf_latent, gpus=gpus_latent, nodes=1, mode='train')
    
    # ---------------------------
    # Step 4: Unconditional Sampling and Evaluation
    # ---------------------------
    print("\nStep 4: Unconditional Sampling and Evaluation")
    gpus_eval = [0, 1, 2, 3]  # Utilize multiple GPUs for efficiency
    conf_eval = ffhq128_autoenc_latent()  # Use the same latent DPM configuration
    conf_eval.eval_programs = ['fid(10,10)']  # Specify the evaluation program (e.g., FID with T=10 and T_latent=10)
    train(conf_eval, gpus=gpus_eval, nodes=1, mode='eval')

if __name__ == '__main__':
    main()

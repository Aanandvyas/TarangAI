from transformers import TimesformerModel

# NO PREPROCESSOR_CONFIG file (UPDATE NEEDED)

# Required for multiprocessing on Windows
if __name__ == "__main__":  
    model_name = "facebook/timesformer-base-finetuned-k400"
    
    print(f"Downloading {model_name}...")
    
    # Load model from Hugging Face
    model = TimesformerModel.from_pretrained(model_name)
    
    # Save the model in a specified directory
    save_path = r"E:\Codes\Models"
    model.save_pretrained(save_path)
    
    print(f"TimeSformer Model Downloaded and Saved Successfully at {save_path}!")
import argparse
import gc
import os
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLaVA vision-language model")
    parser.add_argument("--model_path", type=str, 
                        default=r"F:\TarangAI - 23BCE11755\models\llava-onevision-qwen2-7b-ov-hf",
                        help="Path to the model")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    return parser.parse_args()

def load_model(model_path: str):
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, processor

def generate_response(
    model,
    processor,
    prompt: str,
    image_path: str = None,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    # Process text-only input
    if image_path is None or not os.path.exists(image_path):
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    # Process image + text input
    else:
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return "Error: Could not read the image file."
            
            # Convert BGR to RGB (OpenCV loads as BGR, but models expect RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image with the processor
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        except Exception as e:
            return f"Error processing image: {str(e)}"

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up memory
    del inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return response

def main():
    args = parse_arguments()
    
    model, processor = load_model(args.model_path)
    print("Model loaded successfully!")
    
    print("\nLLaVA-1.5-7B Chat")
    print("Commands: 'exit' to quit, 'image: [path]' to process an image")
    print("=" * 50)
    
    max_length = args.max_length
    temperature = args.temperature
    top_p = args.top_p
    current_image = None
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        # Check if user is providing an image path
        if user_input.lower().startswith("image:"):
            image_path = user_input[6:].strip()
            if os.path.exists(image_path):
                current_image = image_path
                print(f"Image loaded: {image_path}")
            else:
                print(f"Image not found: {image_path}")
            continue
            
        print("\nLLaVA: ", end="", flush=True)
        response = generate_response(
            model, 
            processor, 
            user_input, 
            current_image,
            max_length,
            temperature,
            top_p
        )
        print(response)

if __name__ == "__main__":
    main()
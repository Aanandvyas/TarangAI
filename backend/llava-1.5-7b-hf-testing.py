import argparse
import gc
import os
import time
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from threading import Thread
from typing import Optional, List, Dict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLaVA vision-language model")
    parser.add_argument("--model_path", type=str, 
                        default=r"F:\TarangAI - 23BCE11755\models\llava-1.5-7b-hf",
                        help="Path to the model")
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"], default="fp16",
                        help="Precision for model loading")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--resize", type=int, default=None, 
                        help="Resize image to this size (maintains aspect ratio)")
    return parser.parse_args()

def load_model(model_path: str, precision: str = "fp16"):
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Configure precision settings
    dtype = torch.float16
    quantization_config = None
    
    if precision == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None
    elif precision == "int4":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        dtype = None
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    return model, processor

def preprocess_image(image_path: str, resize: Optional[int] = None):
    """Load and preprocess an image with optional resizing"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if specified
    if resize:
        h, w = image.shape[:2]
        # Calculate new dimensions while preserving aspect ratio
        if h > w:
            new_h, new_w = resize, int(w * resize / h)
        else:
            new_h, new_w = int(h * resize / w), resize
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

def generate_response(
    model,
    processor,
    prompt: str,
    image_path: str = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    resize: Optional[int] = None
) -> str:
    try:
        # Handle conversation history
        if conversation_history is None:
            conversation_history = []
        
        # Process text-only input
        if image_path is None or not os.path.exists(image_path):
            messages = conversation_history + [{"role": "user", "content": prompt}]
            inputs = processor.chat_processor(messages=messages, return_tensors="pt").to(model.device)
        # Process image + text input
        else:
            try:
                # Load and preprocess image
                image = preprocess_image(image_path, resize)
                
                # Create inputs with both image and text
                messages = conversation_history + [{"role": "user", "content": prompt}]
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            except Exception as e:
                return f"Error processing image: {str(e)}"
    
        # Generate response
        with torch.no_grad():
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed_time = time.time() - start_time
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        if image_path is None:
            # For text-only, the output might include the prompt, so extract just the response
            # This is model-specific and might need adjustment
            response = response.split("assistant:")[-1].strip()
        
        # Clean up memory
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Add generation stats
        response += f"\n\n[Generated in {elapsed_time:.2f}s]"
        return response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def show_image(image_path):
    """Display image in a window using OpenCV"""
    if not os.path.exists(image_path):
        print("Image not found")
        return False
        
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read the image")
        return False
    
    # Resize large images for display
    h, w = img.shape[:2]
    max_size = 800
    if h > max_size or w > max_size:
        if h > w:
            new_h, new_w = max_size, int(w * max_size / h)
        else:
            new_h, new_w = int(h * max_size / w), max_size
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    window_name = "Current Image"
    cv2.imshow(window_name, img)
    
    # Create a thread to wait for key press
    def wait_for_key():
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
    Thread(target=wait_for_key).start()
    print("Image displayed. Press any key in the image window to close it.")
    return True

def capture_from_webcam(resize=None):
    """Capture an image from the webcam"""
    print("Opening webcam... Press SPACE to capture or ESC to cancel.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Create window
    window_name = "Webcam Capture"
    cv2.namedWindow(window_name)
    
    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam.")
            break
            
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        
        # Press SPACE to capture
        if key == 32:  # SPACE key
            img = frame.copy()
            break
        # Press ESC to cancel
        elif key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyWindow(window_name)
    
    if img is None:
        return None
    
    # Save the captured image to a temporary file
    temp_path = "webcam_capture.jpg"
    cv2.imwrite(temp_path, img)
    print(f"Image captured and saved to {temp_path}")
    return temp_path

def main():
    args = parse_arguments()
    
    # Load model and processor
    model, processor = load_model(args.model_path, args.precision)
    print("Model loaded successfully!")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)    # Convert to GB
        print(f"Using GPU: {device_name}")
        print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    else:
        print("Using CPU (no GPU detected)")
    
    print("\nLLaVA-1.5-7B Chat")
    print("Commands:")
    print("  'exit' to quit")
    print("  'image: [path]' to load an image")
    print("  'webcam' to capture from webcam")
    print("  'show' to display the current image")
    print("  'clear' to reset conversation history")
    print("-" * 50)
    
    max_length = args.max_length
    temperature = args.temperature
    top_p = args.top_p
    resize = args.resize
    current_image = None
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
            
        # Check if user is providing an image path
        if user_input.lower().startswith("image:"):
            image_path = user_input[6:].strip()
            if os.path.exists(image_path):
                current_image = image_path
                # Reset conversation when new image is loaded
                conversation_history = []
                print(f"Image loaded: {image_path}")
            else:
                print(f"Image not found: {image_path}")
            continue
        
        # Capture from webcam
        elif user_input.lower() == 'webcam':
            image_path = capture_from_webcam(resize)
            if image_path:
                current_image = image_path
                # Reset conversation when new image is loaded
                conversation_history = []
                print(f"Using webcam image: {image_path}")
            continue
            
        # Display the current image
        elif user_input.lower() == 'show':
            if current_image:
                show_image(current_image)
            else:
                print("No image has been loaded yet")
            continue
        
        # Clear conversation history
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("Conversation history cleared.")
            continue
            
        # Generate response
        print("\nLLaVA: ", end="", flush=True)
        response = generate_response(
            model, 
            processor, 
            user_input, 
            current_image,
            conversation_history,
            max_length,
            temperature,
            top_p,
            resize
        )
        print(response)
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history at a reasonable length
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

if __name__ == "__main__":
    main()
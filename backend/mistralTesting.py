import argparse
import gc
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Mistral chat model")
    parser.add_argument("--model_path", type=str, 
                        default=r"F:\TarangAI - 23BCE11755\models\models--mistralai--Mistral-7B-Instruct-v0.3\snapshots\e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
                        help="Path to the model")
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "int4"], default="fp16",
                        help="Precision for model loading")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    return parser.parse_args()

def load_model(model_path: str, precision: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"Loading model with {precision} precision...")
    quantization_config = None
    dtype = torch.float16
    
    if precision == "int16":
        quantization_config = {"load_in_8bit": True}
        dtype = None
    elif precision == "int4":
        quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
        dtype = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config
    )
    
    return model, tokenizer

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    if conversation_history is None:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = conversation_history + [{"role": "user", "content": prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    del inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return response.strip()

def main():
    args = parse_arguments()
    
    model, tokenizer = load_model(args.model_path, args.precision)
    print("Model loaded successfully!")
    
    conversation_history = []
    
    print("\nMistral-7B-Instruct Chat")
    print("Commands: 'exit' to quit, 'clear' to reset conversation, 'params' to modify parameters")
    print("-" * 50)
    
    max_length = args.max_length
    temperature = args.temperature
    top_p = args.top_p
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("Conversation history cleared.")
            continue
        elif user_input.lower() == 'params':
            try:
                max_length = int(input("Enter max length (default 512): ") or max_length)
                temperature = float(input("Enter temperature (default 0.7): ") or temperature)
                top_p = float(input("Enter top_p (default 0.9): ") or top_p)
                print(f"Parameters updated: max_length={max_length}, temperature={temperature}, top_p={top_p}")
            except ValueError:
                print("Invalid input. Using previous parameters.")
            continue
        
        conversation_history.append({"role": "user", "content": user_input})
        
        print("\nMistral: ", end="", flush=True)
        response = generate_response(
            model, 
            tokenizer, 
            user_input, 
            conversation_history[:-1] if len(conversation_history) > 1 else None,
            max_length,
            temperature,
            top_p
        )
        print(response)
        
        conversation_history.append({"role": "assistant", "content": response})
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]

if __name__ == "__main__":
    main()
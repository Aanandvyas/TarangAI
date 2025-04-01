from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

question = "Tell me about Bharatanatyam dance."
response = chatbot(question, max_length=200, do_sample=True)

print(response[0]["generated_text"])
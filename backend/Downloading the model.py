from transformers import TimesformerModel

model_name = "facebook/timesformer-base-finetuned-k400"

print(f"Downloading {model_name}...")
model = TimesformerModel.from_pretrained(model_name)

#Change the path according to your system
model.save_pretrained(r"D:\Scholar\Bhaskar Chari 2022BAI10155\Models")

print("TimeSformer Model Downloaded and Saved Successfully!")
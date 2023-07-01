from Galvatron import GalvatronMega


galvatron = GalvatronMega(use_4bit_quantization=True)
modality = ["Image"]
img_path = "./examples/Tesla-robots.webp"

img_weight = 1.0
response = galvatron.generate(modality, img_path, img_weight, None, 0, None, 0, None, 0)
print(response)
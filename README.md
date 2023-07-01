
# Galvatron: A Multi-Modal Conversational AI

Galvatron is an open-source conversational AI model that leverages the power of MPT30b from MosaicML and uses Multi-Modal adapters to process a variety of input modalities. It's a powerful tool that enables more natural and complex interactions between humans and machines.

## Features
- Uses MPT30b, a highly capable language model developed by MosaicML.
- Supports Multi-Modal input processing with various adapters.
- Aims to facilitate more human-like conversations with AI.

## Usage
```python
from galvatron import Galvatron

model = Galvatron()

response = model.process(input)
print(response)
```

## Roadmap

Our vision for Galvatron is to create an AI model that can understand and generate responses based on a wide array of input modalities. Here's what we're planning to accomplish:

- **Integration with Multi-Modal Adapters:** We're focusing on integrating Galvatron with various multi-modal adapters. This will allow the model to understand and process a wider array of inputs, from text to images and beyond.

- **Fine-tuning on Multi-Modal Instruction Datasets:** Once the integrations are in place, our next goal is to fine-tune Galvatron on multi-modal instruction tuning datasets. This will improve the model's understanding and response generation capabilities across different modalities.

Please check back here for updates on our progress!

## Contributing
We welcome contributions from the community! Please check out the [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to get started.

## License
Galvatron is licensed under [MIT](./LICENSE).

## Acknowledgements
We'd like to acknowledge the teams behind [MPT30b](https://huggingface.co/mosaicml/mpt-30b-chat) and the various multi-modal adapters we're using. Your work has made projects like Galvatron possible!


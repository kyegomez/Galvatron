# Galvatron

Galvatron is a multi-modal conversational model that leverages the power of MPT-30B, and multi-modal adapters to process a variety of input modalities. Built on a robust framework, Galvatron allows you to easily integrate language models with other multi-modal adapters and fine-tune them on multi-modal instruction tuning datasets.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Roadmap](#roadmap)
- [How to Contribute](#how-to-contribute)
- [Share with Friends](#share-with-friends)
- [License](#license)


## Features
- Uses MPT30b, a highly capable language model developed by MosaicML.
- Supports Multi-Modal input processing with various adapters.
- Aims to facilitate more human-like conversations with AI.


## Installation

You can install Galvatron by two methods - using pip, or by cloning the repository from GitHub.

### Pip Install
You can install Galvatron using pip as follows:

```shell
pip install galvatron
```

### Clone the Repository

To clone the Galvatron repository from GitHub, use the following commands:

```shell
git clone https://github.com/kyegomez/Galvatron.git
cd Galvatron
pip install -e .
```

## Quickstart

To get started with Galvatron, please refer to the example given below:

```python
from galvatron import GalvatronBaseLM

galvatron = GalvatronBaseLM(use_4bit_quantization=True)
text = "What is your theory of everything?"
response = galvatron.generate(text)
print(response)
```

## Roadmap

Our current plan for Galvatron's development includes the following:

1. **Integrate the Language Model with other Multi-modal Adapters:** We aim to enhance the versatility of Galvatron by allowing it to process different input modalities effectively.
   
2. **Fine-tune Galvatron on Multi-modal Instruction Tuning Datasets:** To improve its performance and applicability, we plan to fine-tune Galvatron on multi-modal instruction tuning datasets.

We welcome community contributions towards achieving these goals. If you have any suggestions or ideas, please feel free to create an issue or a pull request.

## How to Contribute

We love contributions! If you want to contribute to Galvatron, please fork the repository and create a pull request. If you find any issues or have any feature requests, feel free to create an issue.

## Share with Friends

Help us spread the word about Galvatron. Share this link with your friends and colleagues: [Galvatron on GitHub](https://github.com/kyegomez/Galvatron)

You can also share on social media:

- [Share on Twitter](https://twitter.com/intent/tweet?text=Check+out+Galvatron%2C+a+multi-modal+conversational+model+that+leverages+the+power+of+MPT-30B%2C+and+multi-modal+adapters!&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FGalvatron)
- [Share on LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FGalvatron&title=Galvatron&summary=Galvatron%20is%20a%20multi-modal%20conversational%20model%20that%20leverages%20the%20power%20of%20MPT-30B,%20and%20multi-modal%20adapters.%20Check%20it%20out!)
- [Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FGalvatron)


## License

Galvatron is open-source software, licensed under [MIT](https://github.com/kyegomez/Galvatron/blob/main/LICENSE).


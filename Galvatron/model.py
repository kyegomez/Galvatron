import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

class GalvatronBaseLM:
    """
    This class is designed to initialize the base language model, in this case MosaicML's mpt-30b-chat, 
    with the option for 4-bit quantization. 
    It also integrates additional tools and training efficiency features such as FlashAttention, ALiBi, QK LayerNorm, and more.
    """
    def __init__(self, model_id: str = 'mosaicml/mpt-30b-chat', use_4bit_quantization: bool = False, quant_config: BitsAndBytesConfig = None):
        """
        :param model_id: model identifier used for loading the model from transformers library
        :param use_4bit_quantization: flag indicating whether to use 4-bit quantization
        :param quant_config: an instance of BitsAndBytesConfig for 4-bit quantization
        """
        self.model_id = model_id
        self.use_4bit_quantization = use_4bit_quantization
        self.quant_config = quant_config if quant_config is not None else self.default_quant_config()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'
        config.init_device = 'cuda:0'

        if self.use_4bit_quantization:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config, quantization_config=self.quant_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config, trust_remote_code=True)
    
    @staticmethod
    def default_quant_config():
        """
        Default 4bit quantization configuration.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    def generate(self, text: str, max_new_tokens: int = 100):
        """
        Generate a response from the language model.
        :param text: input text
        :param max_new_tokens: maximum number of new tokens for the generated text
        :return: generated text
        """
        inputs = self.tokenizer(text, return_tensors="pt").to('cuda:0')
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


galvatron = GalvatronBaseLM(use_4bit_quantization=True)
text="What is your theory of everythibg"
response = galvatron.generate(text)
print(response)
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


# galvatron = GalvatronBaseLM(use_4bit_quantization=True)
# text="What is your theory of everythibg"
# response = galvatron.generate(text)
# print(response)

from ImageBind.models import imagebind_model
from ImageBind.models.imagebind_model import ModalityType
from ImageBind.data import data

class Galvatron(GalvatronBaseLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imagebind_model = imagebind_model.imagebind_huge(pretrained=True).eval().to("cuda:0")

    def embed_multi_modal_inputs(self, text: str, image_path: str = None, audio_path: str = None):
        #transform and load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([text], 'cuda:0')
        }

        if image_path is not None:
            inputs[ModalityType.VISION] = data.load_and_transform_vision_data([image_path], 'cuda:0')

        if audio_path is not None:
            inputs[ModalityType.AUDIO] = data.load_and_transform_audio_dataset([audio_path], 'cuda:0')

        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)

        return embeddings
    
    def generate(self, text: str, image_path: str = None, audio_path: str = None, max_new_tokens: int = 100):
        embeddings = self.embed_multi_modal_inputs(text, image_path, audio_path)
        outputs = self.model.generate(embeddings, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    


################# = V3

class GalvatronMega(GalvatronBaseLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imagebind_model = imagebind_model.imagebind_huge(pretrained=True).eval().to('cuda:0')

    def embed_multimodal_inputs(self, modality, img_path, img_weight, text_path, text_weight, video_path, video_weight, audio_path, audio_weight, point_path, point_weight):
        inputs = {}

        if 'Image' in modality:
            image = data.load_and_transform_vision_data([img_path], 'cuda:0')
            inputs['Image'] = [image, img_weight]
        
        if 'Text' in modality:
            text = data.load_and_transform_text([text_path], 'cuda:0')
            inputs['Text'] = [text, text_weight]
        
        if 'Video' in modality:
            video = data.load_and_transform_video_data([video_path], 'cuda:0')
            inputs['Video'] = [video, video_weight]
        
        if 'Audio' in modality:
            audio = data.load_and_transform_audio_data([audio_path], 'cuda:0')
            inputs['Audio'] = [audio, audio_weight]
        
        if 'Point Cloud' in modality:
            point = data.load_and_transform_point_cloud_data([point_path], 'cuda:0')
            inputs['Point Cloud'] = [point, point_weight]
        
        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)
        
        return embeddings

    def generate(self, modality, img_path, img_weight, text_path, text_weight, video_path, video_weight, audio_path, audio_weight, point_path, point_weight, max_new_tokens: int = 100, output_type: str = 'Text'):
        embeddings = self.embed_multimodal_inputs(modality, img_path, img_weight, text_path, text_weight, video_path, video_weight, audio_path, audio_weight, point_path, point_weight)
        
        if output_type == 'Text':
            outputs = self.model.generate(embeddings, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif output_type == 'Image':
            raise NotImplementedError('Image output is not yet implemented')
        
        else:
            raise ValueError('Output type not recognized')


class GalvatronUltra(GalvatronBaseLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imagebind_model = imagebind_model.imagebind_huge(pretrained=True).eval().to("cuda:0")

    def embed_multimodal_inputs(self, modality_data):
        inputs = {}
        load_and_transform = {
            'Image': data.load_and_transform_vision_data,
            'Text': data.load_and_transform_text,
            'Video': data.load_and_transform_video_data,
            'Audio': data.load_and_transform_audio_data,
            'Point Cloud': data.load_and_transform_point_cloud_data,
        }

        for modality, data_path in modality_data.items():
            if data_path is not None:
                transformed_data = load_and_transform[modality]([data_path], 'cuda:0')
                inputs[modality] = transformed_data

        with torch.no_grad():
            embeddings = self.imagebind_model(inputs)

        return embeddings
    
    def generate(self, modality_data, max_new_tokens: int = 100, output_type: str = 'Text'):
        if not isinstance(modality_data, dict):
            raise TypeError("modality_data must be of type dict")

        for modality in modality_data:
            if modality not in ['Image', 'Text', 'Video', 'Audio', 'Point Cloud']:
                raise ValueError(f"Invalid modality: {modality}")

        embeddings = self.embed_multimodal_inputs(modality_data)

        if output_type == 'Text':
            outputs = self.model.generate(embeddings, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif output_type == 'Image':
            raise NotImplemented("Image output is not yet implemented")
        
        else:
            raise ValueError("Output type not recognized")


galvatronMega = GalvatronUltra(use_4bit_quantization=True)
modality_data = {
    "Text": "Text data",
    "Image": "/path/to/image.jpg",
    "Audio": "/path/to/audio.mp3",
    # "Video": "/path/to/video.mp4",  # Uncomment if video data is available
    # "Point Cloud": "/path/to/pointcloud.data",  # Uncomment if point cloud data is available
}
response = galvatronMega.generate(modality_data)
print(response)

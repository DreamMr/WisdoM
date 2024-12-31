import os
from models.basemodel import BaseModel
from models.instructblip import InstructBlipConfig, InstructBlipModel, InstructBlipPreTrainedModel, InstructBlipForConditionalGeneration, InstructBlipProcessor
import torch
from PIL import Image
import transformers
import numpy as np


class MMICL(BaseModel):
    def __init__(self,path,processor_path,model_type="instructblip"):
        self.path = path
        self.model_type = model_type
        self.processor_path = processor_path
        kwargs = {"device_map":"auto"}
        
        config = InstructBlipConfig.from_pretrained(self.path)
        self.decoder_start_token_id = config.text_config.decoder_start_token_id
        if 'instructblip' in model_type:
            model = InstructBlipForConditionalGeneration.from_pretrained(
                self.path,
                config=config
            ).to(dtype=torch.bfloat16)
            self.model = model
            
        image_palceholder="图"
        sp = [image_palceholder] + [f"<image{i}" for i in range(20)]
        processor = InstructBlipProcessor.from_pretrained(
            processor_path
        )
        
        sp = sp + processor.tokenizer.additional_special_tokens[len(sp):]
        processor.tokenizer.add_special_tokens({"additional_special_tokens":sp})
        self.processor = processor
        
        if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
            model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
        self.replace_token="".join(32*[image_palceholder])
        
        self.choice_ids = np.array([
            self.processor.tokenizer.encode('A')[0],
            self.processor.tokenizer.encode('B')[0],
            self.processor.tokenizer.encode('C')[0],
        ])
        #print('Choice ids: ',self.choice_ids)
        
    def score(self,**kwargs):
        pass
    
    def _shift_right(self,input_ids):
        shift_input_ids = torch.zeros((input_ids.shape[0],1))
        shift_input_ids[...,0] = self.decoder_start_token_id
        
        return shift_input_ids.long()
    
    def generate(
        self,
        input_text,
        image_path,
        max_new_tokens=1024,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        num_outputs=1,
        early_stopping=True,
        **kwargs
    ):
        '''
        input_text  e.g. 
        f'image 0 is <image0>{replace_token},image 1 is <image1>{replace_token},image 2 is <image2>{replace_token}. Question: <image0> is a chinchilla. They are mainly found in Chile.\n Question: <image1> is a shiba. They are very popular in Japan.\nQuestion: image 2 is'
        '''
        #print(input_text)
        if isinstance(image_path,str) or isinstance(image_path,list):
            if isinstance(image_path,str):
                image_path_list = [image_path]
            elif isinstance(image_path,list):
                image_path_list = image_path
            image_list = [self.load_image(img_path) for img_path in image_path_list]
        elif image_path is not None:
            image_list = [image_path]
        else:
            image_list = None

        if image_list is not None:
            inputs = self.processor(images=image_list,text=input_text,return_tensors='pt')
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
            inputs['img_mask'] = torch.tensor([[1 for i in range(len(image_list))]])
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        else:
            inputs = self.processor(text=input_text,return_tensors='pt')
        
        decoder_input_ids = self._shift_right(inputs['input_ids']).cuda()
        for k,v in inputs.items():
            inputs[k] = v.cuda()
        
        if image_list is not None:
            logits = self.model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                img_mask=inputs['img_mask'],
                decoder_input_ids=decoder_input_ids
            ).logits
        else:
            print('pixel value is None!')
            logits = self.model(
                pixel_values=None,
                img_mask=None,
                input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                decoder_input_ids=decoder_input_ids
            ).logits
        last_token_logits = logits[:,0]
        prob = torch.nn.functional.softmax(last_token_logits[:, self.choice_ids].float(), dim=1).detach().cpu().numpy().tolist()
        # get prediction in a list like ['A', 'D', 'C', 'B']
        pred = [{0: "A", 1: "B", 2: "C"}[int(num)] for num in np.argmax(prob, axis=1)]
        generated_text = "{}).".format(pred[0])
        return generated_text,prob[0]

    
    def place_holder(self,text):
        text = text.replace("<||image_place_holder||>",f"<image0>{self.replace_token}")
        return text

    def load_image(self,path):
        image = Image.open(path).convert('RGB')
        return image
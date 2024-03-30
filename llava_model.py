import torch
from PIL import Image
from abc import abstractproperty
import os, sys
import os.path as osp
from transformers import AutoTokenizer, AutoModel
import time
import warnings

class LLaVA:

    INSTALL_REQ = True

    def __init__(self, 
                 model_pth,
                 conque_pth=None,
                 **kwargs): 
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn("Please install llava before using LLaVA")
            sys.exit(-1)
            
        assert osp.exists(model_pth) or len(model_pth.split('/')) == 2
        
        model_name = get_model_name_from_path(model_pth)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_pth, 
                model_base=None, 
                model_name=model_name, 
                device='cpu', 
                device_map='cpu'
            )
        except Exception as e:
            print(e)
            if 'ShareGPT4V' in model_pth:
                import llava
                warnings.warn(
                    f'Please manually remove the encoder type check in {llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)
        
        self.model = self.model.cuda()
        self.conv_mode =  'llava_v1'

        kwargs_default = dict(do_sample=True, temperature=0.2, max_new_tokens=1024, top_p=0.5, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
    
    def logit_generate(self,image_path,prompt):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        if isinstance(image_path,str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        question = prompt['question']
        options = prompt['options']
        
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        question_input_ids = tokenizer_image_token(prompt,self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).cuda()
        
        output_question = self.model(
            question_input_ids,
            use_cache=True,
            images=image_tensor,
            return_dict=True
        )
        question_logits = output_question.logits
        question_past_key_values = output_question.past_key_values
        
        loss_list = []
        logit_list = {}
        chosen_list = []
        for option_k,option_v in options.items():
            chosen_list.append(option_k)

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], option_v)
            option_prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(option_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            option_answer_input_ids = input_ids[:,question_input_ids.shape[1]:]
            with torch.inference_mode():
                output_option = self.model(
                    input_ids=option_answer_input_ids,
                    use_cache=True,
                    attention_mask=torch.ones(1,question_logits.shape[1]+option_answer_input_ids.shape[1]).cuda(),
                    past_key_values=question_past_key_values,
                    return_dict=True
                )
                logits = torch.cat([question_logits[:,-1:],output_option.logits[:,:-1]],1)
                prob_logit = torch.nn.functional.softmax(logits,dim=-1).view(-1,self.model.config.vocab_size)
                loss_fct = torch.nn.CrossEntropyLoss()
                logits = logits.view(-1,self.model.config.vocab_size)
                labels = option_answer_input_ids.view(-1)
                loss = loss_fct(logits,labels)
                loss_list.append(-loss)
        
        logit_prob = torch.nn.functional.softmax(torch.tensor(loss_list)).detach().cpu().numpy()
        option_chosen = torch.stack(loss_list).argmax().item()
        
        return {'answer':chosen_list[option_chosen],'logit_score':logit_prob}

    def generate(self, image_path, prompt):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        
        if isinstance(image_path,str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.batch_decode(output_ids,skip_special_tokens=True)[0].strip()
        return output

from llava_model import LLaVA
import numpy as np

def contextual_fusion(prob_ori, prob_context, alpha, beta):
    """
    :param prob_ori: numpy 1d array-like, the probability of original
    :param prob_context: numpy 1d array-like, the probability of incorporating context
    :param alpha: float, the threshold of choosing hard sample
    :param beta: float, interpolation coefficient
    """
    delta = 2 * np.max(prob_ori) + np.min(prob_ori) - 1
    if delta > alpha:
        return prob_ori
    
    prob_fuse = prob_ori + beta * (prob_context - prob_ori)
    
    return prob_fuse


if __name__ == '__main__':
    # model path
    lvlm_path = r'llava-v1.5-13b'
    msa_path = r'YOUR MODEL PATH'
    
    # prompt template
    TEMPLATE = {
        "stage2": """Give you an image and sentence, you can provide historical context, important events, and relevant background information related to the image and sentence. Sentence: \"{}\"""",
        "question": """Sentence: \"{}\" Use the image as a visual aids to help you answer the question. What is the sentiment polarity of the aspect \"{}\" in the sentence?\nA). positive\nB). neutral\nC). negative\nAnswer with the option's letter from the given choices directly.""",
        "question_context": """Sentence: \"{}\". Context: \"{}\". Use the image as a visual aids to help you answer the question. What is the sentiment polarity of the aspect \"{}\" in the sentence?\nA). positive\nB). neutral\nC). negative\nAnswer with the option's letter from the given choices directly.""",
        "output":"""
Sentence: {}
Aspect: {}
Ground Truth: {}
Context: {}
Original Answer: {}, probability: {}
+ Context: {}, probability: {}
+ Contextual Fusion: {}, probability: {}"""
    }
    
    # initialize model
    lvlm = LLaVA(lvlm_path) # LVLM model for stage 2
    msa = LLaVA(msa_path) # msa model for stage 3
    
    img_path = r'demo.jpg'
    sentence = "Good evening , Boston"
    aspect = "Boston"
    
    # stage 2 Context Generation
    inp = TEMPLATE['stage2'].format(sentence)
    context = lvlm.generate(img_path,inp)
    
    # stage 3 Contextual Fusion
    question = TEMPLATE['question'].format(sentence,aspect)
    options = {"A). positive":"A). positive","B). neutral":"B). neutral","C). negative":"C). negative"}
    prompt = {"question": question, "options": options}
    
    dic_ori = msa.logit_generate(img_path,prompt) # Original prediction
    
    question = TEMPLATE['question_context'].format(sentence,context,aspect)
    prompt = {"question": question, "options": options}
    dic_context = msa.logit_generate(img_path,prompt) # Prediction with context
    
    prob_final = contextual_fusion(dic_ori['logit_score'],dic_context['logit_score'],alpha=0.3,beta=0.4) # Fusion
    
    options_list = ['A). positive','B). neutral','C). negative']
    chosen_idx = np.argmax(prob_final)
    
    print(TEMPLATE['output'].format(
        sentence,
        aspect,
        "A). positive",
        context,
        dic_ori['answer'], "{{positive: {}, neutral: {}, negative: {}}}".format(dic_ori['logit_score'][0],dic_ori['logit_score'][1],dic_ori['logit_score'][2]),
        dic_context['answer'], "{{positive: {}, neutral: {}, negative: {}}}".format(dic_context['logit_score'][0],dic_context['logit_score'][1],dic_context['logit_score'][2]),
        options_list[chosen_idx],"{{positive: {}, neutral: {}, negative: {}}}".format(prob_final[0],prob_final[1],prob_final[2])
    ))
    
    
    

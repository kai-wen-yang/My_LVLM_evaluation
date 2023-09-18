import torch
from omegaconf import OmegaConf
import cheetah
from cheetah.common.config import Config
from cheetah.common.registry import registry
from cheetah.conversation.conversation import Chat, CONV_VISION

from cheetah.models import *
from cheetah.processors import *
from . import get_image
import pdb


class TestCheetah:
    def __init__(self) -> None:
        config = OmegaConf.load("/fs/nexus-scratch/kwyang3/My_LVLM_evaluation/models/cheetah/cheetah_eval_vicuna.yaml")
        cfg = Config.build_model_config(config)
        model_cls = registry.get_model_class(cfg.model.arch)
        self.model = model_cls.from_config(cfg.model).cuda()
        vis_processor_cfg = cfg.preprocess.vis_processor.eval
        self.vis_processors = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.chat = Chat(self.model, self.vis_processors, device=self.device)


    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image, "prompt": question}, max_length=max_new_tokens)[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        raw_img_list = []
        for image in image_list:
            raw_img_list.append([image])

        context = ["<ImageHere> "+ques for ques in question_list]
        output = self.chat.batch_answer(raw_img_list, context, max_new_tokens=max_new_tokens)
        # outputs = []
        # for image, question in zip(image_list, question_list):
        #     output = self.chat.answer([image], "<Img><HereForImage></Img> "+question, max_new_tokens=max_new_tokens)
        #     outputs.append(output)
        return outputs

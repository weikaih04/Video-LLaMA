import argparse
import os

from .common.config import Config
from .common.registry import registry
from .conversation.conversation_video import Chat, default_conversation, conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

from .datasets.builders import *
from .models import *
from .processors import *
from .runners import *
from .tasks import *


def create_args(cfg_path='eval_configs/video_llama_eval_withaudio.yaml', model_type='LLama2', torch_device='cuda:0'):
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='LLama2', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    
    # Manually create the args object.
    # Namespace is the type returned by parse_args() and is suitable for holding arguments.
    args = argparse.Namespace(
        cfg_path=cfg_path,
        gpu_id=int(torch_device.split(':')[-1]) if torch_device.startswith('cuda:') else 0,  # Extract GPU ID from device string
        model_type=model_type,
        options=[],  # Default empty list, adjust as needed
    )

    return args


class ChatBot:

    def __init__(self, cfg_path, llama_model_path, vl_model_path, model_type, torch_device):
        args = create_args(cfg_path=cfg_path, model_type=model_type, torch_device=model_type)
        self.chat = self._init_model(args, llama_model_path, vl_model_path)
        if args.model_type == 'vicuna':
            self.chat_state = default_conversation.copy()
        else:
            self.chat_state = conv_llava_llama_2.copy()
        self.img_list = list()
        self.set_para()

    def _init_model(self, args, llama_model_path, vl_model_path):
        print('Initializing Chat')
        cfg = Config(args, llama_model_path, vl_model_path)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        print('Initialization Finished')
        return chat

    def set_para(self, num_beams=1, temperature=1.0):
        self.num_beams = num_beams
        self.temperature = temperature
        print('set num_beams: {}'.format(num_beams))
        print('set temperature: {}'.format(temperature))

    def upload(self, up_img=False, up_video=False, audio_flag=False):
        if up_img and not up_video:
            self.chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            self.chat.upload_img(up_img, self.chat_state, self.img_list)
        elif not up_img and up_video:
            self.chat_state.system =  ""
            if audio_flag:
                self.chat.upload_video(up_video, self.chat_state, self.img_list)
            else:
                self.chat.upload_video_without_audio(up_video, self.chat_state, self.img_list)

    def ask_answer(self, user_message):
        self.chat.ask(user_message, self.chat_state)
        llm_message = self.chat.answer(conv=self.chat_state,
                                       img_list=self.img_list,
                                       num_beams=self.num_beams,
                                       temperature=self.temperature,
                                       max_new_tokens=300,
                                       max_length=2000)[0]

        return llm_message
        

    def reset(self):
        if self.chat_state is not None:
            self.chat_state.messages = list()
        if self.img_list is not None:
            self.img_list = list()
        self.set_para()


if __name__ == "__main__":

    args = parse_args()
    chatbot = ChatBot(args)

    while True:
        try:
            file_path = input('Input file path: ')
        except:
            print('Input error, try again.')
            continue
        else:
            if file_path == 'exit':
                print('Goodbye!')
                break
            if not os.path.exists(file_path):
                print('{} not exist, try again.'.format(file_path))
                continue

        # chatbot.upload(up_img=file_path)
        chatbot.upload(up_video=file_path, audio_flag=True)

        while True:
            try:
                user_message = input('User: ')
            except:
                print('Input error, try again.')
                continue
            else:
                if user_message == 'para':
                    num_beams = int(input('Input new num_beams:(1-10) '))
                    temperature = float(input('Input new temperature:(0.1-2.0) '))
                    chatbot.set_para(num_beams=num_beams, temperature=temperature)
                    continue
                if user_message == 'exit':
                    break
            
            llm_message = chatbot.ask_answer(user_message=user_message)
            print('ChatBot: {}'.format(llm_message))

        chatbot.reset()
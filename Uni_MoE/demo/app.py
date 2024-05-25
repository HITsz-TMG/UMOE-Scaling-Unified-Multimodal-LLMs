import os
import gradio as gr
import requests
import json

if __name__ == '__main__':
    # gr.close_all()
    title = "Welcome to Uni-MoE"
    description = "<center><font size=5> Scaling Unified Multimodal LLMs with Mixture of Experts</font></center>"

    prompt_input = gr.Textbox(label="Prompt:", placeholder="Give your prompt here.", lines=2)
    image_input = gr.Image(type="filepath")
    audio_input = gr.Audio(source =  "microphone", type="filepath")
    video_input = gr.Video()

    # device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # print(device)

    print("Loading...")

    print("Success!")

    def predict(prompt_input, image_input, audio_input, video_input):
        flag = {"image":image_input,
                    "audio":audio_input,
                    "video":video_input}
        files = {"image":(image_input,open(image_input,'rb')) if image_input is not None else None,
                    "audio":(audio_input,open(audio_input,'rb')) if audio_input is not None else None,
                    "video":(video_input,open(video_input,'rb')) if video_input is not None else None
                    }
        data = {"query":prompt_input,"flag":flag}
        print(data)
        headers = {'content-type': 'multipart/form-data'}
        print(image_input)
        # url should be the server of the demo_speech
        res = requests.post(url="http://10.196.16.110:9011/gen",  data={"data":json.dumps(data)},files=files).json()
        output = res["output"]
        print(output)
        # if image_input is not None: os.remove(image_input)
        # if audio_input is not None: os.remove(audio_input)
        # if video_input is not None: os.remove(video_input)
        return str(output)

    gr.Interface(
        fn=predict,
        inputs=[prompt_input, image_input, audio_input, video_input],
        outputs="text",
        title=title,
        description=description,
        allow_flagging="never",
        examples=[
            ["Describe the following image in detail."],
            ["Give an elaborate explanation of the image you see."],
            ["Render a thorough depiction of this chart."],
            ["Narrate the contents of the image with precision."],
            ["Illustrate the image through a descriptive explanation."],
            ["Introduce me this painting in detail."],
            ["Provide an elaborate account of this painting."],
            ["Outline a detailed portrayal of this diagram."],
            ["Provide an elaborate account of this chart."],
            ["Give detailed answer for this question. Question: "]
        ]
    ).launch(share=False,ssl_verify=False, ssl_certfile="E:/path/to/cert.pem", ssl_keyfile="E:/path/to/key.pem")
    # must use https to ensure the microphone input
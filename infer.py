import os
import torch
import gradio as gr
from PIL import Image
from pathlib import Path
from prismatic import load
import warnings
import shutil

# --- 0. [新增] 修复代理问题 ---
# 设置 NO_PROXY 环境变量，告诉系统访问本地地址时不要使用代理。
# 这是解决 Gradio 启动时因代理干扰而报 403 错误的关键。
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

# --- 1. 配置和环境变量 ---
os.environ['HF_HUB_OFFLINE'] = '1'
warnings.filterwarnings("ignore")

try:
    HF_TOKEN = Path(".hf_token").read_text().strip()
except FileNotFoundError:
    HF_TOKEN = "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" # 请替换为你的Hugging Face Token

# --- 2. 模型和设备定义 ---
MODEL_ID_1 = "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/svd_siglip/prism-svd_dual_siglip_ft_proj_noalign_qwen"
MODEL_ID_2 = "/mnt/world_foundational_model/gkz/prismatic-vlms/runs/original_weight/reproduction-llava-v15+7b"

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    print("错误：此应用需要至少两个可用的 CUDA GPU 设备。")
    exit()

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# --- 3. 模型加载 ---
print("="*50)
print("正在加载模型，请稍候...")

print(f"在设备 {device0} 上加载模型 1: {MODEL_ID_1.split('/')[-1]}")
vlm1 = load(MODEL_ID_1, hf_token=HF_TOKEN)
vlm1.to(device0, dtype=torch.bfloat16)
print("模型 1 加载完成。")

print(f"在设备 {device1} 上加载模型 2: {MODEL_ID_2.split('/')[-1]}")
vlm2 = load(MODEL_ID_2, hf_token=HF_TOKEN)
vlm2.to(device1, dtype=torch.bfloat16)
print("模型 2 加载完成。")

print("所有模型已成功加载。Gradio UI 正在启动...")
print("="*50)

# --- 4. 推理函数 (已修改) ---
def compare_models(pil_images, user_prompt, output_dir):
    """接收多张图片和提示，调用两个模型进行推理。"""
    if not pil_images:
        return "错误：请上传至少一张图片。", "错误：请上传至少一张图片。"
    if not user_prompt or not user_prompt.strip():
        return "错误：请输入有效的提示。", "错误：请输入有效的提示。"
    if not output_dir or not output_dir.strip():
        return "错误：请输入有效的预览图保存路径。", "错误：请输入有效的预览图保存路径。"

    # [修复] Gradio Gallery 可能会返回 (image, caption) 元组，我们需要正确提取图像对象。
    images = []
    for item in pil_images:
        img = item[0] if isinstance(item, tuple) else item
        images.append(img.convert("RGB"))

    # [新增] 创建或清空预览图输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    # --- 模型 1 推理 (支持多图) ---
    # 获取图片张量用于预览
    image_transform = vlm1.vision_backbone.image_transform
    pixel_values = image_transform(images)
    if isinstance(pixel_values, torch.Tensor):
        # 如果只有一张图，需要增加一个batch维度
        if len(pixel_values.shape) == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(vlm1.device)
    elif isinstance(pixel_values, dict):
        # 适配字典类型的pixel_values
        processed_pixel_values = {}
        for k, v in pixel_values.items():
            # 同样检查并增加batch维度
            if len(v.shape) == 3:
                v = v.unsqueeze(0)
            processed_pixel_values[k] = v.to(vlm1.device)
        pixel_values = processed_pixel_values
    else:
        raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    # 提取用于预览的特定张量
    svd_pixel_values = pixel_values.get("svd", pixel_values)
    svd_pixel_values = svd_pixel_values[None, ...]
    print("处理过的图片形状：", svd_pixel_values.shape)
    vlm1.vision_backbone.preview(svd_pixel_values, output_dir=output_dir)

    # 为模型 1 生成结果
    prompt_builder1 = vlm1.get_prompt_builder()
    prompt_builder1.add_turn(role="human", message=user_prompt)
    prompt_text1 = prompt_builder1.get_prompt()
    generated_text1 = vlm1.generate(
        images, prompt_text1, do_sample=True, temperature=0.1, max_new_tokens=1024, min_length=1
    )
    generated_text1 += f"\n\n[提示] 水印预览图已保存至: {output_dir}"


    # --- 模型 2 推理 (仅支持单图) ---
    # LLaVA 模型只处理第一张图片
    llava_image = images[0]
    prompt_builder2 = vlm2.get_prompt_builder()
    prompt_builder2.add_turn(role="human", message=user_prompt)
    prompt_text2 = prompt_builder2.get_prompt()
    generated_text2 = vlm2.generate(
        llava_image, prompt_text2, do_sample=True, temperature=0.1, max_new_tokens=1024, min_length=1
    )

    # 如果输入是多张图片，向模型2的输出添加提示信息
    if len(images) > 1:
        generated_text2 += "\n\n[注意] LLaVA 模型仅支持单图输入，以上结果仅针对您上传的 *第一张* 图片。"

    return generated_text1, generated_text2

# --- 5. Gradio UI 界面 (已修改) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 视觉语言模型 (VLM) 对比工具
    上传一张或多张图片，输入一个问题或指令，然后点击 "开始比较" 按钮。

    **模型1 (Device: {device0})**: `{MODEL_ID_1.split('/')[-1]}` (支持多图)
    **模型2 (Device: {device1})**: `{MODEL_ID_2.split('/')[-1]}` (仅使用第一张图)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # [修改] 使用 gr.Gallery 支持多图上传
            image_input = gr.Gallery(label="上传图片 (可多张)", type="pil", elem_id="gallery")
            prompt_input = gr.Textbox(label="输入提示 (Prompt)", value="What will happen?", lines=3)
            # [新增] 允许用户指定输出目录
            output_dir_input = gr.Textbox(label="水印预览图保存路径", value="./wm_output", lines=1)
            submit_btn = gr.Button("开始比较", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                output1 = gr.Textbox(label="模型 1 输出", lines=15, interactive=False, container=False)
                output2 = gr.Textbox(label="模型 2 输出", lines=15, interactive=False, container=False)

    # [修改] 更新 click 事件的输入参数
    submit_btn.click(fn=compare_models, inputs=[image_input, prompt_input, output_dir_input], outputs=[output1, output2])

    gr.Examples(
        examples=[
            [[os.path.abspath("./assets/beignets.png")], "Describe this image in detail.", "./wm_output_dir"],
            [[os.path.abspath("./assets/rocket.png")], "What is in this image and what is going on?", "./wm_output_dir"],
        ],
        inputs=[image_input, prompt_input, output_dir_input],
        outputs=[output1, output2],
        fn=compare_models,
        cache_examples=False,
    )

# --- 6. 启动应用 ---
if __name__ == "__main__":
    # 尝试创建并下载示例图片，如果失败则打印警告
    try:
        if not os.path.exists("assets"):
            os.makedirs("assets")
        import requests

        urls = {
            "beignets.png": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
            "rocket.png": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/rocket.png" # 添加一个新的示例图片
        }

        for filename, url in urls.items():
            filepath = os.path.join("assets", filename)
            if not os.path.exists(filepath):
                print(f"正在下载示例图片: {filename}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status() # 如果下载失败则抛出异常
                with open(filepath, "wb") as f:
                    f.write(response.content)
    except Exception as e:
        print("\n" + "*"*50)
        print(f"警告：无法下载示例图片。这可能是由于网络代理或防火墙问题。")
        print(f"错误详情: {e}")
        print("应用将继续启动，但示例可能无法点击。您可以手动上传图片。")
        print("*"*50 + "\n")

    # 启动 Gradio 应用
    demo.launch(share=True)
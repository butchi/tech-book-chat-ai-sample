import os
import openai
from ipykernel.kernelbase import Kernel
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv('.env')

client = openai.OpenAI()

class ChatKernel(Kernel):
    implementation = 'GPT Image Custom'
    implementation_version = '1.1.1'
    language = 'no-op'
    language_version = '0.1'
    language_info = {
        'name': 'Any text',
        'mimetype': 'text/plain',
        'file_extension': '.txt',
    }
    banner = "Chat kernel - openai API with image generation custom"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = os.environ.get("GPT_IMAGE_MODEL")
        self.size = os.environ.get("GPT_IMAGE_SIZE")
        self.quality = os.environ.get("GPT_IMAGE_QUALITY")

    def generate_image(self, prompt):
        openai.api_key = self.api_key
        try:
            response = client.images.generate(
                model=self.model,
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                n=1,
            )
            return response.data[0].b64_json
        except Exception as e:
            return f"Error: {str(e)}"

    # カーネル情報を表示するメソッド
    def show_kernel_info(self):
        self.send_response(self.iopub_socket, 'display_data', {
            'data': {'text/plain': [
                f"{self.implementation} v{self.implementation_version}",
                f"current model: {self.model}",
                f"current size: {self.size}",
                f"current quality: {self.quality}"
            ]},
            'metadata': {}
        })

    def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False
    ):
        code = code.strip()
        if not code:
            return {
                'status': 'ok',
                'execution_count': self.execution_count,
                'payload': [],
                'user_expressions': {},
            }

        content = self.generate_image(code)

        # 初回実行時にカーネル情報を表示
        if self.execution_count == 1:
            self.show_kernel_info()

        if not silent:
            if content.startswith("Error:"):
                self.send_response(self.iopub_socket, 'display_data', {
                    'data': {'text/markdown': content},
                    'metadata': {}
                })
            else:
                self.send_response(self.iopub_socket, 'display_data', {
                    'data': {'image/png': content},
                    'metadata': {}
                })

        return {
            'status': 'ok',
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {},
        }

if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=ChatKernel)

import json
import os
import threading
from langchain_openai import ChatOpenAI
import shutil
import subprocess
import re


class Translate_LLM:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = ChatOpenAI(
                    model='mistral-nemo:12b-instruct-2407-fp16',
                    api_key='mistral-nemo:12b-instruct-2407-fp16',
                    openai_api_base="http://112.48.199.7:11434/v1/"
                )
        return cls._instance


def chunk_text_old(content):
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 chunk.js 的绝对路径
    chunk_js_path = os.path.join(current_dir, 'chunk.js')

    # 将内容写入临时文件
    temp_file_path = os.path.join(current_dir, 'temp_content_all.txt')
    with open(temp_file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # 构建临时输出文件路径
    temp_output_file = os.path.join(current_dir, 'temp_chunks.json')

    # 执行 chunk.js,将结果输出到临时文件
    subprocess.run(['node', chunk_js_path, temp_file_path, temp_output_file], check=True)

    # 从临时文件读取结果
    with open(temp_output_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"获取到 {len(chunks)} 个 chunk")

    # 删除临时文件
    os.remove(temp_file_path)
    os.remove(temp_output_file)

    return chunks


def chunk_text(content, max_chunk_size=1000):
    chunks = []
    current_chunk = ""
    lines = content.split('\n')

    in_code_block = False
    in_list = False
    list_indent = 0
    in_table = False
    in_blockquote = False

    for line in lines:
        stripped_line = line.strip()
        
        # 检查是否进入或退出代码块
        if stripped_line.startswith('```'):
            in_code_block = not in_code_block
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += line + '\n'
            continue

        # 检查是否在表格中
        if not in_code_block:
            if stripped_line.startswith('|') and '-|-' in stripped_line:
                in_table = True
            elif in_table and not stripped_line.startswith('|'):
                in_table = False

        # 检查是否在引用块中
        if stripped_line.startswith('>'):
            in_blockquote = True
        elif in_blockquote and not stripped_line.startswith('>'):
            in_blockquote = False

        # 检查是否在列表中
        if not in_code_block and not in_table:
            list_match = re.match(r'^(\s*)[-*+]\s', line) or re.match(r'^(\s*)\d+\.\s', line)
            if list_match:
                current_indent = len(list_match.group(1))
                if not in_list or current_indent > list_indent:
                    in_list = True
                    list_indent = current_indent
            elif in_list and (not stripped_line or len(line) - len(line.lstrip()) <= list_indent):
                in_list = False
                list_indent = 0

        # 如果当前行是标题、在代码块中、在列表中、在表格中、在引用块中，或者当前块加上这行会超过最大大小，就开始新的块
        if (stripped_line.startswith('#') or
                in_code_block or
                in_list or
                in_table or
                in_blockquote or
                len(current_chunk) + len(line) > max_chunk_size):
            if current_chunk and not in_code_block and not in_table:
                chunks.append(current_chunk.strip())
                current_chunk = ""

        current_chunk += line + '\n'

    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def translate_chunk(chunk, target_language, max_attempts=4):
    llm = Translate_LLM()
    prompt = f"""
    请将以下Markdown格式的文本翻译成-{target_language}语言,保持原文的格式和结构不变。
    翻译时请注意以下几点：

    1. 保留所有的Markdown语法,包括标题、列表、链接、代码块等。
    2. 保持原文的段落结构和换行。
    3. 不要翻译代码块内的内容。
    4. 专有名词和技术术语应保持其原有形式或使用公认的翻译。
    5. 翻译应当准确、流畅,符合目标语言的表达习惯。

    以下是需要翻译的文本：

    {chunk}

    JSON格式返回翻译结果,不需要多余描述,格式如下：
    {{"translated_text": "仅仅翻译后的文本内容"}}
    """
    for attempt in range(max_attempts):
        response = llm.invoke(prompt)

        try:
            # 尝试直接解析整个响应内容为JSON
            result = json.loads(response.content)
            if "translated_text" in result:
                return result["translated_text"]
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if "translated_text" in result:
                        return result["translated_text"]
                except json.JSONDecodeError:
                    pass

        print(f"尝试 {attempt + 1}/{max_attempts}: 未能获取有效的翻译结果，重试中...")

    print(f"经过 {max_attempts} 次尝试后仍未获得有效的JSON响应，返回原始响应内容")
    return response.content


def translate_md_file(input_file, output_folder, target_language):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割内容为chunks
    chunks = chunk_text(content)

    # 翻译每个chunk
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"正在翻译第 {i + 1}/{len(chunks)} 个chunk...")
        translated_chunk = translate_chunk(chunk, target_language)
        translated_chunks.append(translated_chunk)

        # 输出原文和译文
        print(f"\n原文:\n{chunk}")
        print(f"\n译文:\n{translated_chunk}")
        print("-" * 50)

        # 生成临时输出文件名
        temp_dir = os.path.join(output_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_output_filename = f"translated_{os.path.basename(input_file)}_progress_{i + 1}.md"
        temp_output_path = os.path.join(temp_dir, temp_output_filename)

        # 将当前进度保存到临时文件
        with open(temp_output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(translated_chunks))

        print(f"当前进度已保存到: {temp_output_path}")

    # 合并翻译后的chunks
    translated_content = "\n".join(translated_chunks)

    # 生成输出文件名
    output_filename = f"translated_{os.path.basename(input_file)}"
    output_path = os.path.join(output_folder, output_filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将翻译后的内容写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(translated_content)

    # 删除临时文件夹
    temp_dir = os.path.join(output_folder, 'temp')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"临时文件夹已删除: {temp_dir}")

    print(f"翻译完成。翻译后的文件已保存到: {output_path}")


if __name__ == "__main__":
    input_file = "../content.md"
    output_folder = "../translated_files"
    target_language = "英语"  # 可以根据需要更改目标语言

    translate_md_file(input_file, output_folder, target_language)

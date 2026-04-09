<<<<<<< Updated upstream
# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
=======
from text_representation import *

DATA_PATH = "TicTacToe_Data/train/text"

sentences = load_sentences(DATA_PATH)

print("Example sentence:")
print(sentences[0])

vocab = build_vocab(sentences)

print("\nVocabulary size:")
print(len(vocab))

encoded = encode_sentence(sentences[0], vocab)

print("\nEncoded sentence:")
print(encoded)

padded = pad_sequence(encoded, 40, vocab["<pad>"])

print("\nPadded sentence:")
print(padded)
>>>>>>> Stashed changes
